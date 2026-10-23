package gpuq

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

const (
	fixtureMode  = "GPUQ_FD9_TEST_FIXTURE"
	fixtureReady = "GPUQ_FD9_TEST_READY"
	fixtureStop  = "GPUQ_FD9_TEST_STOP"
)

func TestMain(m *testing.M) {
	switch os.Getenv(fixtureMode) {
	case "parent":
		if err := spawnLockHolder(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		os.Exit(0)
	case "holder":
		signal.Ignore(syscall.SIGTERM)
		for {
			if _, err := os.Stat(os.Getenv(fixtureStop)); err == nil {
				os.Exit(0)
			}
			time.Sleep(10 * time.Millisecond)
		}
	}
	os.Exit(m.Run())
}

func TestRunnerTerminatesDescendantAndReleasesQueueLease(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("the GPUQ lock descriptor regression test requires Linux")
	}

	packageDir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	cli := filepath.Join(packageDir, "gpuq.rb")
	testBinary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	readyPath := filepath.Join(dir, "holder.ready")
	stopPath := filepath.Join(dir, "holder.stop")
	lockPath := filepath.Join(dir, "queue.lock")
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "ruby", cli, "run", "--name", "fd9-descendant", "--exclude", "gpu,engine,trace,profiler,provenance", "--", testBinary, "-test.run=^TestFixtureNoop$")
	cmd.Env = envWith(os.Environ(), map[string]string{
		"GPUQ_DB":      filepath.Join(dir, "history.sqlite3"),
		"GPUQ_LOCK":    lockPath,
		"GPUQ_QUEUE":   filepath.Join(dir, "queue"),
		"GPUQ_JOB_ID":  "",
		"GPUQ_LOCK_FD": "",
		fixtureMode:    "parent",
		fixtureReady:   readyPath,
		fixtureStop:    stopPath,
	})
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	done := make(chan error, 1)
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	go func() { done <- cmd.Wait() }()

	var childPID int
	deadline := time.Now().Add(10 * time.Second)
	for childPID == 0 && time.Now().Before(deadline) {
		if data, err := os.ReadFile(readyPath); err == nil {
			childPID, _ = strconv.Atoi(strings.TrimSpace(string(data)))
		}
		if childPID != 0 {
			break
		}
		select {
		case runErr := <-done:
			t.Fatalf("gpuq exited before the lock-holding child started: %v\nstdout: %s\nstderr: %s", runErr, stdout.String(), stderr.String())
		default:
		}
		time.Sleep(10 * time.Millisecond)
	}
	if childPID == 0 {
		t.Fatalf("the lock-holding child did not start\nstdout: %s\nstderr: %s", stdout.String(), stderr.String())
	}
	childReleased := false
	defer func() {
		if !childReleased {
			_ = os.WriteFile(stopPath, []byte("stop"), 0600)
			_ = syscall.Kill(childPID, syscall.SIGKILL)
		}
	}()

	select {
	case runErr := <-done:
		if runErr != nil {
			t.Fatalf("gpuq run failed: %v\nstdout: %s\nstderr: %s", runErr, stdout.String(), stderr.String())
		}
	case <-ctx.Done():
		t.Fatalf("gpuq did not return after its command leader exited; recorded status: %s\nstdout: %s\nstderr: %s", recordedStatus(cli, cmd.Env), stdout.String(), stderr.String())
	}

	if status := recordedStatus(cli, cmd.Env); status != "capture_failed" {
		t.Fatalf("expected a recorded descendant capture failure, got %q", status)
	}
	probe, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		t.Fatal(err)
	}
	defer probe.Close()
	deadline = time.Now().Add(2 * time.Second)
	for {
		err := syscall.Flock(int(probe.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
		if err == nil {
			_ = syscall.Flock(int(probe.Fd()), syscall.LOCK_UN)
			break
		}
		if err != syscall.EWOULDBLOCK && err != syscall.EAGAIN {
			t.Fatalf("unexpected result waiting for descendant to release the queue lease: %v", err)
		}
		if time.Now().After(deadline) {
			t.Fatal("queue lease remained held after bounded descendant cleanup")
		}
		time.Sleep(10 * time.Millisecond)
	}
	childReleased = true
}

func TestFixtureNoop(t *testing.T) {}

func spawnLockHolder() error {
	usageFD, err := syscall.Dup(4)
	if err != nil {
		return fmt.Errorf("runner did not pass timing pipe descriptor 4: %w", err)
	}
	usage := os.NewFile(uintptr(usageFD), "gpuq-timing-pipe")
	defer usage.Close()
	lockFD, err := syscall.Dup(9)
	if err != nil {
		return fmt.Errorf("runner did not pass descriptor 9: %w", err)
	}
	lock := os.NewFile(uintptr(lockFD), "gpuq-lock")
	defer lock.Close()
	null, err := os.Open(os.DevNull)
	if err != nil {
		return err
	}
	defer null.Close()

	binary, err := os.Executable()
	if err != nil {
		return err
	}
	cmd := exec.Command(binary, "-test.run=^TestFixtureNoop$")
	cmd.Env = envWith(os.Environ(), map[string]string{fixtureMode: "holder"})
	cmd.Stdout, cmd.Stderr = io.Discard, io.Discard
	cmd.ExtraFiles = []*os.File{null, usage, null, null, null, null, lock}
	if err := cmd.Start(); err != nil {
		return err
	}
	if err := os.WriteFile(os.Getenv(fixtureReady), []byte(strconv.Itoa(cmd.Process.Pid)), 0600); err != nil {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return err
	}
	return cmd.Process.Release()
}

func recordedStatus(cli string, env []string) string {
	query := exec.Command("ruby", cli, "query", "SELECT status FROM jobs WHERE label = 'fd9-descendant' ORDER BY submitted_ns DESC LIMIT 1")
	query.Env = env
	data, err := query.Output()
	if err != nil {
		return fmt.Sprintf("query failed: %v", err)
	}
	var rows []map[string]string
	if err := json.Unmarshal(data, &rows); err != nil {
		return fmt.Sprintf("invalid query result: %v", err)
	}
	if len(rows) == 0 {
		return "missing"
	}
	return rows[0]["status"]
}

func envWith(base []string, values map[string]string) []string {
	env := make([]string, 0, len(base)+len(values))
	for _, entry := range base {
		key, _, _ := strings.Cut(entry, "=")
		if _, replace := values[key]; !replace {
			env = append(env, entry)
		}
	}
	for key, value := range values {
		env = append(env, key+"="+value)
	}
	return env
}
