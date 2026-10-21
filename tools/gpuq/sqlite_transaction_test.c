#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <signal.h>
#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static void child_fail(const char *where, int rc, sqlite3 *db) {
    fprintf(stderr, "%s failed: rc=%d (%s)\n", where, rc,
            db ? sqlite3_errmsg(db) : "no database");
    _exit(1);
}

static void write_byte(int fd, char value) {
    ssize_t written;
    do {
        written = write(fd, &value, 1);
    } while (written < 0 && errno == EINTR);
    if (written != 1) {
        _exit(1);
    }
}

static char read_byte(int fd) {
    char value;
    ssize_t received;
    do {
        received = read(fd, &value, 1);
    } while (received < 0 && errno == EINTR);
    return received == 1 ? value : '\0';
}

static sqlite3 *open_db(const char *path) {
    sqlite3 *db = NULL;
    int rc = sqlite3_open_v2(path, &db,
                             SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, NULL);
    if (rc != SQLITE_OK) {
        child_fail("sqlite3_open_v2", rc, db);
    }
    sqlite3_extended_result_codes(db, 1);
    sqlite3_busy_timeout(db, 5000);
    return db;
}

static void exec_sql(sqlite3 *db, const char *sql, const char *where) {
    char *error = NULL;
    int rc = sqlite3_exec(db, sql, NULL, NULL, &error);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "%s failed: rc=%d (%s)\n", where, rc,
                error ? error : sqlite3_errmsg(db));
        sqlite3_free(error);
        _exit(1);
    }
}

static void initialize_db(const char *path) {
    sqlite3 *db = open_db(path);
    exec_sql(db,
             "PRAGMA journal_mode=WAL;"
             "CREATE TABLE documents ("
             "id INTEGER PRIMARY KEY, job_id TEXT NOT NULL, kind TEXT NOT NULL,"
             "source TEXT NOT NULL, ordinal INTEGER NOT NULL, observed_ns INTEGER NOT NULL,"
             "bytes INTEGER NOT NULL, sha256 TEXT NOT NULL, codec TEXT NOT NULL, data BLOB NOT NULL,"
             "UNIQUE(job_id, kind, source, ordinal));"
             "CREATE TABLE metrics (id INTEGER PRIMARY KEY, value TEXT NOT NULL);",
             "initialize database");
    if (sqlite3_close(db) != SQLITE_OK) {
        child_fail("close initialized database", SQLITE_ERROR, db);
    }
}

static void select_document_identity(sqlite3 *db) {
    sqlite3_stmt *statement = NULL;
    const char *sql =
        "SELECT id, sha256 FROM documents "
        "WHERE job_id=? AND kind=? AND source=? AND ordinal=?";
    int rc = sqlite3_prepare_v2(db, sql, -1, &statement, NULL);
    if (rc != SQLITE_OK) {
        child_fail("prepare document identity SELECT", rc, db);
    }
    sqlite3_bind_text(statement, 1, "job", -1, SQLITE_STATIC);
    sqlite3_bind_text(statement, 2, "artifact", -1, SQLITE_STATIC);
    sqlite3_bind_text(statement, 3, "stdout", -1, SQLITE_STATIC);
    sqlite3_bind_int(statement, 4, 0);
    rc = sqlite3_step(statement);
    while (rc == SQLITE_ROW) {
        rc = sqlite3_step(statement);
    }
    sqlite3_finalize(statement);
    if (rc != SQLITE_DONE) {
        child_fail("read document identity", rc, db);
    }
}

static int insert_document(sqlite3 *db) {
    sqlite3_stmt *statement = NULL;
    const char *sql =
        "INSERT INTO documents(job_id,kind,source,ordinal,observed_ns,bytes,sha256,codec,data) "
        "VALUES('job','artifact','stdout',0,1,1,'test-hash','raw',x'58')";
    int rc = sqlite3_prepare_v2(db, sql, -1, &statement, NULL);
    if (rc == SQLITE_OK) {
        rc = sqlite3_step(statement);
    }
    if (rc != SQLITE_DONE) {
        rc = sqlite3_extended_errcode(db);
    }
    sqlite3_finalize(statement);
    return rc;
}

static void insert_metric(sqlite3 *db) {
    exec_sql(db, "INSERT INTO metrics(value) VALUES('writer');", "insert metric");
}

static int wait_child(pid_t child, const char *name) {
    int status;
    pid_t waited;
    do {
        waited = waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    if (waited != child) {
        perror("waitpid");
        return 0;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "%s child failed\n", name);
        return 0;
    }
    return 1;
}

static int test_deferred_snapshot(const char *path) {
    int ready[2];
    int release[2];
    if (pipe(ready) || pipe(release)) {
        perror("pipe");
        return 0;
    }
    pid_t child = fork();
    if (child < 0) {
        perror("fork");
        return 0;
    }
    if (child == 0) {
        close(ready[0]);
        close(release[1]);
        sqlite3 *db = open_db(path);
        exec_sql(db, "BEGIN DEFERRED;", "begin deferred transaction");
        select_document_identity(db);
        write_byte(ready[1], 'R');
        if (read_byte(release[0]) != 'C') {
            child_fail("wait for competing writer", SQLITE_ERROR, db);
        }
        int rc = insert_document(db);
        if (rc != SQLITE_BUSY_SNAPSHOT) {
            fprintf(stderr,
                    "deferred transaction expected SQLITE_BUSY_SNAPSHOT, got %d (%s)\n",
                    rc, sqlite3_errmsg(db));
            _exit(1);
        }
        exec_sql(db, "ROLLBACK;", "rollback stale snapshot");
        sqlite3_close(db);
        _exit(0);
    }

    close(ready[1]);
    close(release[0]);
    if (read_byte(ready[0]) != 'R') {
        fprintf(stderr, "deferred transaction did not finish its identity SELECT\n");
        return 0;
    }
    sqlite3 *writer = open_db(path);
    exec_sql(writer, "BEGIN IMMEDIATE;", "begin competing writer");
    insert_metric(writer);
    exec_sql(writer, "COMMIT;", "commit competing writer");
    sqlite3_close(writer);
    write_byte(release[1], 'C');
    close(ready[0]);
    close(release[1]);
    if (!wait_child(child, "deferred snapshot")) {
        return 0;
    }
    puts("PASS deferred read snapshot reproduces SQLITE_BUSY_SNAPSHOT on document INSERT");
    return 1;
}

struct BusyNotice {
    int fd;
    int notified;
};

static int notify_while_busy(void *context, int attempts) {
    struct BusyNotice *notice = context;
    if (!notice->notified) {
        write_byte(notice->fd, 'B');
        notice->notified = 1;
    }
    return attempts < 5000;
}

static void write_immediate_document(const char *path, int ready_fd, int release_fd) {
    sqlite3 *db = open_db(path);
    exec_sql(db, "BEGIN IMMEDIATE;", "begin immediate document transaction");
    select_document_identity(db);
    write_byte(ready_fd, 'R');
    if (read_byte(release_fd) != 'C') {
        child_fail("wait for immediate transaction release", SQLITE_ERROR, db);
    }
    if (insert_document(db) != SQLITE_OK) {
        child_fail("insert document in immediate transaction", sqlite3_errcode(db), db);
    }
    exec_sql(db, "COMMIT;", "commit immediate document transaction");
    sqlite3_close(db);
    _exit(0);
}

static void write_waiting_metric(const char *path, int started_fd) {
    sqlite3 *db = open_db(path);
    struct BusyNotice notice = {.fd = started_fd, .notified = 0};
    sqlite3_busy_handler(db, notify_while_busy, &notice);
    write_byte(started_fd, 'A');
    exec_sql(db, "BEGIN IMMEDIATE;", "begin waiting metric transaction");
    insert_metric(db);
    exec_sql(db, "COMMIT;", "commit waiting metric transaction");
    sqlite3_close(db);
    _exit(0);
}

static int test_immediate_serialization(const char *path) {
    int ready[2];
    int release[2];
    int started[2];
    if (pipe(ready) || pipe(release) || pipe(started)) {
        perror("pipe");
        return 0;
    }
    pid_t document_writer = fork();
    if (document_writer < 0) {
        perror("fork document writer");
        return 0;
    }
    if (document_writer == 0) {
        close(ready[0]);
        close(release[1]);
        close(started[0]);
        close(started[1]);
        write_immediate_document(path, ready[1], release[0]);
    }
    close(ready[1]);
    close(release[0]);
    if (read_byte(ready[0]) != 'R') {
        fprintf(stderr, "immediate transaction did not finish its identity SELECT\n");
        return 0;
    }

    pid_t metric_writer = fork();
    if (metric_writer < 0) {
        perror("fork metric writer");
        return 0;
    }
    if (metric_writer == 0) {
        close(ready[0]);
        close(release[1]);
        write_waiting_metric(path, started[1]);
    }
    close(started[1]);
    if (read_byte(started[0]) != 'A' || read_byte(started[0]) != 'B') {
        fprintf(stderr, "second writer did not block behind BEGIN IMMEDIATE\n");
        return 0;
    }
    write_byte(release[1], 'C');
    close(ready[0]);
    close(release[1]);
    close(started[0]);
    if (!wait_child(document_writer, "immediate document") ||
        !wait_child(metric_writer, "waiting metric")) {
        return 0;
    }
    puts("PASS BEGIN IMMEDIATE serializes the document and metric writers without losing either");
    return 1;
}

static void remove_database(const char *path) {
    char sidecar[512];
    unlink(path);
    if (snprintf(sidecar, sizeof(sidecar), "%s-wal", path) > 0) {
        unlink(sidecar);
    }
    if (snprintf(sidecar, sizeof(sidecar), "%s-shm", path) > 0) {
        unlink(sidecar);
    }
}

int main(void) {
    char directory[] = "/tmp/gpuq-sqlite-transactions-XXXXXX";
    char deferred_db[512];
    char immediate_db[512];
    alarm(15);
    if (!mkdtemp(directory)) {
        perror("mkdtemp");
        return 1;
    }
    if (snprintf(deferred_db, sizeof(deferred_db), "%s/deferred.sqlite", directory) < 0 ||
        snprintf(immediate_db, sizeof(immediate_db), "%s/immediate.sqlite", directory) < 0) {
        fprintf(stderr, "database path formatting failed\n");
        return 1;
    }
    initialize_db(deferred_db);
    initialize_db(immediate_db);
    int ok = test_deferred_snapshot(deferred_db) &&
             test_immediate_serialization(immediate_db);
    remove_database(deferred_db);
    remove_database(immediate_db);
    rmdir(directory);
    return ok ? 0 : 1;
}
