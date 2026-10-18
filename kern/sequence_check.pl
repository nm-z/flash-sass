#!/usr/bin/env perl
use strict;
use warnings;
use 5.016;
use Getopt::Long qw(GetOptions);
use IO::Socket::UNIX;
use IO::Select;
use Socket qw(SOCK_STREAM SHUT_RDWR);
use JSON::PP;
use Time::HiRes qw(clock_gettime CLOCK_MONOTONIC sleep);
use File::Temp qw(tempfile);
use File::Basename qw(dirname);
use Digest::SHA;
use Errno qw(EINTR EAGAIN EWOULDBLOCK);
$SIG{PIPE} = q{IGNORE};

# Completed GPU turns establish scheduling and cancellation order.
# Client timestamps are evidence, but do not decide those assertions.
my $case = 'all';
my $revision = '';
my $self_test = 0;
my $timeout = $ENV{FLASH_SEQUENCE_TIMEOUT_S} // 300;
GetOptions(
    'case=s' => \$case, 'revision=s' => \$revision,
    'timeout=f' => \$timeout, 'self-test' => \$self_test,
) or die "bad options\n";

sub mono { return clock_gettime(CLOCK_MONOTONIC); }
sub need { my ($ok, $message) = @_; die "$message\n" unless $ok; }

sub done_event {
    my ($events) = @_;
    my @done = grep { $_->{event} eq 'done' } @$events;
    need(@done == 1, 'request must have exactly one done trace record');
    return $done[0];
}

sub last_completed_turn {
    my ($events) = @_;
    my $done = done_event($events);
    my @turns = sort { $a->{event_seq} <=> $b->{event_seq} }
                grep { $_->{event} eq 'turn' } @$events;
    need(@turns > 0, 'request has no completed GPU turns');
    my $last = $turns[-1];
    need(defined($done->{last_turn_seq}), 'done trace lacks last_turn_seq');
    need($done->{last_turn_seq} == $last->{event_seq},
         'done.last_turn_seq does not identify the final GPU turn');
    need($done->{count} == $last->{count},
         'done count differs from the final completed GPU turn');
    return ($done, $last);
}

sub progress_witness {
    my ($partial, $a_events, $b_events) = @_;
    need($partial->{event} eq 'turn' && !$partial->{aborted}
         && $partial->{count} > 0
         && $partial->{count} < $partial->{nprompt} - 1,
         'long request was not observed partway through prefill');
    my ($bdone, $bfinal) = last_completed_turn($b_events);
    need(!$bdone->{aborted}, 'short request was aborted');
    need($bdone->{count} >= $bdone->{total} || $bdone->{ended},
         'short request did not finish');
    my @boundary = sort { $a->{event_seq} <=> $b->{event_seq} }
                   grep { $_->{event} eq 'turn'
                          && $_->{count} >= $_->{nprompt} - 1 } @$a_events;
    need(@boundary > 0, 'long request never reached its prefill boundary');
    need($partial->{event_seq} < $bfinal->{event_seq},
         'short request finished before the observed long prefill started');
    need($bfinal->{event_seq} < $boundary[0]{event_seq},
         'short request did not finish during the long request prefill');
    return {
        long_partial_turn => $partial->{event_seq},
        long_partial_count => $partial->{count},
        short_final_turn => $bfinal->{event_seq},
        short_final_count => $bfinal->{count},
        long_prefill_boundary_turn => $boundary[0]{event_seq},
        long_prefill_boundary_count => $boundary[0]{count},
    };
}

sub cancellation_witness {
    my ($partial, $events, $quantum) = @_;
    need($partial->{event} eq 'turn' && !$partial->{aborted}
         && $partial->{count} > 0
         && $partial->{count} < $partial->{nprompt} - 1,
         'cancellation did not follow an observed partial prefill turn');
    my ($done, $last) = last_completed_turn($events);
    need($done->{aborted}, 'disconnected request did not report cancellation');
    need($done->{count} >= $partial->{count},
         'cancelled request count moved backwards');
    need($done->{count} < $done->{total},
         'cancelled request ran to its requested cap');
    need($done->{count} < $done->{nprompt} - 1,
         'disconnected request continued through its entire prefill');
    # A turn may be running when the prior completed turn is observed.
    # Allow another scheduling interval and the kernel's abort lead.
    my $bound = 2 * $quantum + 8;
    need($done->{count} - $partial->{count} <= $bound,
         'cancellation exceeded the documented completed-position bound');
    return {
        observed_turn => $partial->{event_seq},
        observed_count => $partial->{count},
        final_turn => $last->{event_seq}, final_count => $done->{count},
        total => $done->{total}, aborted => JSON::PP::true,
        extra_positions => $done->{count} - $partial->{count},
        maximum_extra_positions => $bound,
    };
}

sub predicate_self_test {
    my @results;
    my $test = sub {
        my ($name, $should_pass, $code) = @_;
        my $ok = eval { $code->(); 1 };
        my $error = $@;
        my $passed = (!!$ok == !!$should_pass);
        push @results, {
            name => $name,
            passed => $passed ? JSON::PP::true : JSON::PP::false,
            expected_acceptance => $should_pass ? JSON::PP::true : JSON::PP::false,
            predicate_error => $error,
        };
    };
    my $turn = sub {
        my ($seq, $count, $nprompt, $total, $aborted) = @_;
        return {
            event => 'turn', event_seq => $seq, count => $count,
            nprompt => $nprompt, total => $total, aborted => $aborted,
        };
    };
    my $done = sub {
        my ($seq, $last, $count, $nprompt, $total, $aborted) = @_;
        return {
            event => 'done', event_seq => $seq, last_turn_seq => $last,
            count => $count, nprompt => $nprompt, total => $total,
            aborted => $aborted, ended => 0,
        };
    };
    my $partial = $turn->(10, 8, 100, 108, 0);
    my @a_good = ($partial, $turn->(30, 99, 100, 108, 0));
    my @b_good = ($turn->(20, 4, 2, 4, 0), $done->(40, 20, 4, 2, 4, 0));
    $test->('causal interleaving accepted despite delayed done', 1, sub {
        progress_witness($partial, \@a_good, \@b_good);
    });
    my @a_serial = ($partial, $turn->(15, 99, 100, 108, 0));
    $test->('serial scheduler rejected', 0, sub {
        progress_witness($partial, \@a_serial, \@b_good);
    });
    my @b_first = ($turn->(5, 4, 2, 4, 0), $done->(6, 5, 4, 2, 4, 0));
    $test->('short request scheduled entirely first rejected', 0, sub {
        progress_witness($partial, \@a_good, \@b_first);
    });
    my @cancel_good = ($partial, $turn->(12, 16, 100, 108, 1),
                       $done->(13, 12, 16, 100, 108, 1));
    $test->('bounded active cancellation accepted', 1, sub {
        cancellation_witness($partial, \@cancel_good, 8);
    });
    my @ignore_cancel = ($partial, $turn->(12, 108, 100, 108, 0),
                         $done->(13, 12, 108, 100, 108, 0));
    $test->('ignored cancellation rejected', 0, sub {
        cancellation_witness($partial, \@ignore_cancel, 8);
    });
    my @late_cancel = ($partial, $turn->(12, 100, 100, 108, 1),
                       $done->(13, 12, 100, 100, 108, 1));
    $test->('cancellation deferred until output rejected', 0, sub {
        cancellation_witness($partial, \@late_cancel, 8);
    });
    my @unbounded_cancel = ($partial, $turn->(12, 48, 100, 108, 1),
                            $done->(13, 12, 48, 100, 108, 1));
    $test->('unbounded cancellation rejected', 0, sub {
        cancellation_witness($partial, \@unbounded_cancel, 8);
    });
    my $failed = grep { !$_->{passed} } @results;
    print JSON::PP->new->canonical->pretty->encode({
        kind => 'checker predicate self-test; no GPU execution',
        passed => $failed ? JSON::PP::false : JSON::PP::true,
        checks => \@results,
    });
    exit($failed ? 1 : 0);
}

predicate_self_test() if $self_test;
need($case =~ /\A(?:all|progress|cancellation)\z/, 'invalid --case');
need($revision =~ /\A[0-9a-f]{40}(?:-mutant-(?:serial|cancellation))?\z/,
     '--revision must be a full committed SHA or its explicit mutant label');
need($timeout > 0, '--timeout must be positive');
need(@ARGV == 3, 'usage: sequence_check.pl --case all|progress|cancellation '
     . '--revision SHA SOCKET TRACE_JSONL REPORT.json');

my ($socket_path, $trace_path, $report_path) = @ARGV;
my $json = JSON::PP->new->canonical;
my $selector = IO::Select->new;
my @requests;
my (%by_fd, %by_request_id, %trace_by_request);
my @trace_events;
my $trace_fh;
my $trace_buffer = '';
my $trace_last_seq = 0;
my $started = mono();
my ($config, $info, $capacity, $quantum);
my $report = {
    schema => 2, case => $case, expected_revision => $revision,
    socket => $socket_path, trace_path => $trace_path,
    timeout_seconds => $timeout,
    checks => [], fixtures => [], result => 'running',
};

sub record_check {
    my ($name, $ok, $details) = @_;
    push @{$report->{checks}}, {
        name => $name, passed => $ok ? JSON::PP::true : JSON::PP::false,
        details => $details,
    };
    need($ok, $name);
}

sub send_all {
    my ($socket, $data) = @_;
    while (length $data) {
        my $n = syswrite($socket, $data);
        next if !defined($n) && $! == EINTR;
        need(defined($n) && $n > 0, "socket write failed: $!");
        substr($data, 0, $n, '');
    }
}

sub connect_socket {
    my $s;
    {
        local $SIG{ALRM} = sub { die "socket connection timed out\n"; };
        alarm(int($timeout) + 1);
        $s = IO::Socket::UNIX->new(Type => SOCK_STREAM, Peer => $socket_path);
        alarm(0);
    }
    need(defined($s), "cannot connect to $socket_path: $!");
    return $s;
}

sub query_line {
    my ($command) = @_;
    my $s = connect_socket();
    send_all($s, "$command\n");
    my $line;
    {
        local $SIG{ALRM} = sub { die "$command timed out\n"; };
        alarm(int($timeout) + 1);
        $line = <$s>;
        alarm(0);
    }
    close($s);
    need(defined($line), "$command closed without a response");
    need($line !~ /\Aerror:/, "$command: $line");
    return $line;
}

sub associate_queued {
    my ($event) = @_;
    return unless $event->{event} eq 'queued';
    for my $r (@requests) {
        next if defined $r->{request_id};
        next unless $event->{event_seq} > $r->{trace_after};
        next unless $event->{wanted} == $r->{slot};
        next unless $event->{nprompt} == @{$r->{ids}};
        next unless $event->{total} == @{$r->{ids}} + $r->{new};
        next if exists $by_request_id{$event->{request_id}};
        $r->{request_id} = $event->{request_id};
        $by_request_id{$event->{request_id}} = $r;
        return;
    }
    die "unattributed request in private trace: " . $json->encode($event) . "\n";
}

sub pump_trace {
    for (;;) {
        my $chunk;
        my $n = sysread($trace_fh, $chunk, 65536);
        next if !defined($n) && $! == EINTR;
        need(defined($n), "cannot read trace: $!");
        last if $n == 0;
        $trace_buffer .= $chunk;
        while ($trace_buffer =~ s/\A([^\n]*)\n//) {
            my $line = $1;
            next unless length $line;
            my $e = eval { $json->decode($line) };
            need(ref($e) eq 'HASH', "invalid trace JSON: $line");
            for my $key (qw(event event_seq request_id wanted slot nprompt total)) {
                need(exists($e->{$key}), "trace lacks $key: $line");
            }
            need($e->{event_seq} > $trace_last_seq,
                 'trace event_seq is not strictly increasing');
            $trace_last_seq = $e->{event_seq};
            push @trace_events, $e;
            push @{$trace_by_request{$e->{request_id}}}, $e;
            associate_queued($e);
        }
    }
}

sub close_request {
    my ($r, $cancel) = @_;
    return unless $r->{socket};
    my $s = delete $r->{socket};
    my $fd = fileno($s);
    $selector->remove($s);
    delete $by_fd{$fd} if defined $fd;
    shutdown($s, SHUT_RDWR) if $cancel;
    close($s);
    $r->{closed_at_seconds} = mono() - $started;
    $r->{client_disconnected} = JSON::PP::true if $cancel;
}

sub parse_stream_line {
    my ($r, $line) = @_;
    push @{$r->{lines}}, $line;
    if ($line =~ /\Atok\s+(\d+)\s*\z/) {
        push @{$r->{tokens}}, 0 + $1;
        return;
    }
    if ($line =~ /\Adone\s/) {
        my @a = split ' ', $line;
        need(@a >= 9, "malformed done record in $r->{label}: $line");
        need(!exists($r->{done}), "duplicate done record in $r->{label}");
        $r->{done} = {
            generated => 0 + $a[1], ended => 0 + $a[2],
            nprompt => 0 + $a[3], run_ms => 0 + $a[4],
            first_ms => 0 + $a[5], resumed => 0 + $a[6],
            snapshot => 0 + $a[7], slot => 0 + $a[8],
        };
        need($r->{done}{generated} == @{$r->{tokens}},
             "token count mismatch in $r->{label}");
        need($r->{done}{slot} == $r->{slot}, "wrong slot in $r->{label}");
        $r->{protocol_done_at_seconds} = mono() - $started;
        # A serial control holds its permit until the client closes.
        close_request($r, 0);
        return;
    }
    die "$r->{label}: $line" if $line =~ /\Aerror:/;
}

sub tick {
    pump_trace();
    for my $s ($selector->can_read(0)) {
        my $fd = fileno($s);
        my $r = $by_fd{$fd};
        next unless $r;
        my $chunk;
        my $n = sysread($s, $chunk, 16384);
        next if !defined($n)
            && ($! == EINTR || $! == EAGAIN || $! == EWOULDBLOCK);
        need(defined($n), "read failed for $r->{label}: $!");
        if ($n == 0) {
            $r->{eof} = JSON::PP::true;
            close_request($r, 0);
            need(exists($r->{done}), "$r->{label} closed without done");
            next;
        }
        $r->{raw_stream} .= $chunk;
        $r->{buffer} .= $chunk;
        while ($r->{buffer} =~ s/\A([^\n]*\n)//) {
            parse_stream_line($r, $1);
        }
    }
    pump_trace();
}

sub wait_for {
    my ($predicate, $description) = @_;
    my $end = mono() + $timeout;
    for (;;) {
        tick();
        my $answer = $predicate->();
        return $answer if $answer;
        need(mono() < $end, "timed out waiting for $description");
        sleep(0.005);
    }
}

sub request_events {
    my ($r) = @_;
    return [] unless defined $r->{request_id};
    return $trace_by_request{$r->{request_id}} // [];
}

sub find_event {
    my ($r, $name) = @_;
    for my $e (@{request_events($r)}) {
        return $e if $e->{event} eq $name;
    }
    return undef;
}

sub start_request {
    my (%p) = @_;
    pump_trace();
    my $s = connect_socket();
    my $r = {
        label => $p{label}, slot => $p{slot}, ids => [@{$p{ids}}],
        new => exists($p{new}) ? $p{new} : 6,
        sample_seed => $p{sample}, snapshot_requested => $p{snap} // 0,
        image => $p{image}, stop => $p{stop},
        trace_after => $trace_last_seq,
        started_at_seconds => mono() - $started,
        raw_stream => '', buffer => '', lines => [], tokens => [], socket => $s,
    };
    push @requests, $r;
    $by_fd{fileno($s)} = $r;
    $selector->add($s);
    my @commands;
    push @commands, "sample $p{sample} .7 8 .9" if defined $p{sample};
    push @commands, "stop $p{stop}" if defined $p{stop};
    push @commands, "snap $p{snap}" if $p{snap};
    if (defined $p{image}) {
        need($p{image} !~ /\s/, 'image protocol cannot carry a spaced path');
        push @commands, "img $p{image} 1";
    }
    push @commands, join(' ', 'chat_slot', $r->{slot}, $r->{new}, @{$r->{ids}});
    $r->{command_stream} = join("\n", @commands) . "\n";
    # Do not wait synchronously for setup acknowledgements: B must be
    # submitted while A is active, including in the serial negative control.
    send_all($s, $r->{command_stream});
    return $r;
}

sub finish_request {
    my ($r) = @_;
    wait_for(sub { exists($r->{done}) }, "$r->{label} protocol completion");
    wait_for(sub { find_event($r, 'done') }, "$r->{label} trace completion");
    my ($d) = last_completed_turn(request_events($r));
    need(!$d->{aborted}, "$r->{label} unexpectedly aborted");
    need($d->{rc} == 0, "$r->{label} trace reported an error");
    need($d->{count} == $d->{total} || $d->{ended},
         "$r->{label} trace did not complete");
    need($d->{slot} == $r->{slot}, "$r->{label} trace slot mismatch");
    need($d->{resumed} == $r->{done}{resumed}
         && $d->{snapshot} == $r->{done}{snapshot},
         "$r->{label} snapshot trace/protocol mismatch");
    close_request($r, 0);
    return $r;
}

sub ask { return finish_request(start_request(@_)); }

sub equal_tokens {
    my ($name, $actual, $expected) = @_;
    my $a = join(',', @{$actual->{tokens}});
    my $b = join(',', @{$expected->{tokens}});
    record_check($name, $a eq $b, {
        actual => $actual->{label}, reference => $expected->{label},
        token_count => scalar @{$actual->{tokens}},
    });
}

sub pair_requests {
    my ($a, $b) = @_;
    my $ra = start_request(%$a);
    my $rb = start_request(%$b);
    finish_request($ra);
    finish_request($rb);
    my @ta = grep { $_->{event} eq 'turn' } @{request_events($ra)};
    my @tb = grep { $_->{event} eq 'turn' } @{request_events($rb)};
    need(@ta && @tb, 'interleaved pair lacks GPU turn evidence');
    record_check("$ra->{label} / $rb->{label}: interleaved GPU turns",
        $ta[0]{event_seq} < $tb[-1]{event_seq}
        && $tb[0]{event_seq} < $ta[-1]{event_seq}, {
            a_first => $ta[0]{event_seq}, a_last => $ta[-1]{event_seq},
            b_first => $tb[0]{event_seq}, b_last => $tb[-1]{event_seq},
        });
    return ($ra, $rb);
}

sub await_partial {
    my ($r) = @_;
    return wait_for(sub {
        for my $e (@{request_events($r)}) {
            return $e if $e->{event} eq 'turn' && !$e->{aborted}
                && $e->{count} > 0 && $e->{count} < $e->{nprompt} - 1;
        }
        need(!find_event($r, 'done'),
             "$r->{label} finished without an observable partial prefill");
        return undef;
    }, "$r->{label} first completed prefill turn");
}

sub long_prompt {
    my $n = $capacity - 16;
    need($n >= 6 * $quantum + 16,
         'slot too small for decisive prefill/cancellation tests at this quantum');
    return [map { 100 + ($_ % 100) } 0 .. $n - 1];
}

sub parity_and_snapshots {
    my @a = (1 .. 8);
    my @b = (65 .. 74);
    my @c = @a;
    $c[0] = 31;
    for my $mode ('greedy', 'sampled') {
        # A two-token mismatched request invalidates prior snapshots, making
        # the following serial references cold even after the greedy suite.
        for my $slot (0, 1) {
            ask(slot => $slot, ids => [900, 901], new => 0,
                label => "$mode clear prior snapshot slot $slot");
        }
        my $seed_a = $mode eq 'sampled' ? 12345 : undef;
        my $seed_b = $mode eq 'sampled' ? 98765 : undef;
        my %pa = (slot => 0, ids => \@a, sample => $seed_a, new => 6);
        my %pb = (slot => 1, ids => \@b, sample => $seed_b, new => 6);
        my $sa = ask(%pa, label => "$mode serial A");
        my $sb = ask(%pb, label => "$mode serial B");
        my $sc = ask(%pa, ids => \@c, label => "$mode serial changed A");
        record_check("$mode cold nonempty serial outputs",
            @{$sa->{tokens}} > 0 && @{$sb->{tokens}} > 0
            && !$sa->{done}{resumed} && !$sb->{done}{resumed}
            && !$sc->{done}{resumed}, {});
        record_check("$mode distinct prompts affect output",
            join(',', @{$sa->{tokens}}) ne join(',', @{$sb->{tokens}})
            && join(',', @{$sa->{tokens}}) ne join(',', @{$sc->{tokens}}), {});
        my ($ia, $ib) = pair_requests(
            {%pa, label => "$mode interleaved A"},
            {%pb, label => "$mode interleaved B"});
        equal_tokens("$mode A parity", $ia, $sa);
        equal_tokens("$mode B parity", $ib, $sb);
        my ($xa, $xb) = pair_requests(
            {%pa, slot => 1, label => "$mode swapped A"},
            {%pb, slot => 0, label => "$mode swapped B"});
        equal_tokens("$mode swapped A parity", $xa, $sa);
        equal_tokens("$mode swapped B parity", $xb, $sb);
        my ($snap_a, $snap_b) = pair_requests(
            {%pa, snap => 2, label => "$mode snapshot A at 2"},
            {%pb, snap => 3, label => "$mode snapshot B at 3"});
        equal_tokens("$mode snapshot A output", $snap_a, $sa);
        equal_tokens("$mode snapshot B output", $snap_b, $sb);
        record_check("$mode unequal snapshots created",
            $snap_a->{done}{snapshot} == 2 && $snap_b->{done}{snapshot} == 3, {});
        my ($res_a, $res_b) = pair_requests(
            {%pa, label => "$mode resume A"},
            {%pb, label => "$mode resume B"});
        equal_tokens("$mode resumed A parity", $res_a, $sa);
        equal_tokens("$mode resumed B parity", $res_b, $sb);
        record_check("$mode independent resume lengths",
            $res_a->{done}{resumed} == 2 && $res_b->{done}{resumed} == 3, {});
        my ($bad_a, $keep_b) = pair_requests(
            {%pa, ids => \@c, label => "$mode invalidate only A"},
            {%pb, label => "$mode preserve B snapshot"});
        equal_tokens("$mode changed A cold parity", $bad_a, $sc);
        equal_tokens("$mode unaffected B parity", $keep_b, $sb);
        record_check("$mode snapshot invalidation isolated",
            $bad_a->{done}{resumed} == 0 && $keep_b->{done}{resumed} == 3, {});
        if ($mode eq 'greedy') {
            my $stopped = ask(%pa, stop => $sa->{tokens}[0],
                              label => 'stop token terminates exactly once');
            record_check('stop token behavior',
                $stopped->{done}{ended} == 1 && @{$stopped->{tokens}} == 1
                && $stopped->{tokens}[0] == $sa->{tokens}[0], {});
        }
    }
}

sub progress_case {
    my $a = start_request(label => 'progress long A', slot => 0,
                          ids => long_prompt(), new => 1);
    my $partial = await_partial($a);
    my $b = start_request(label => 'progress short B', slot => 1,
                          ids => [77, 78], new => 1);
    finish_request($b);
    finish_request($a);
    my $w = progress_witness($partial, request_events($a), request_events($b));
    record_check('short request finished inside observed long prefill', 1, $w);
}

sub cancellation_case {
    my @reuse_ids = (211 .. 218);
    my $baseline = ask(label => 'cancellation reuse serial baseline',
        slot => 0, ids => \@reuse_ids, new => 6);
    my $a = start_request(label => 'active cancellation long A',
        slot => 0, ids => long_prompt(), new => 1);
    my $partial = await_partial($a);
    $a->{disconnect_after_event_seq} = $partial->{event_seq};
    $a->{disconnect_after_count} = $partial->{count};
    close_request($a, 1);
    wait_for(sub { find_event($a, 'done') }, 'active cancellation disposition');
    my $w = cancellation_witness($partial, request_events($a), $quantum);
    record_check('active cancellation stops unfinished prefill within bound', 1, $w);
    my $other = ask(label => 'other slot after active cancellation',
        slot => 1, ids => \@reuse_ids, new => 6);
    equal_tokens('other slot remains correct after cancellation', $other, $baseline);
    my $reuse = ask(label => 'same slot after active cancellation',
        slot => 0, ids => \@reuse_ids, new => 6);
    equal_tokens('cancelled slot reuse matches serial baseline', $reuse, $baseline);

    my $holder = start_request(label => 'queued cancellation slot holder',
        slot => 0, ids => long_prompt(), new => 1);
    my $held_partial = await_partial($holder);
    my $waiter = start_request(label => 'cancelled explicit-slot waiter',
        slot => 0, ids => [301, 302, 303], new => 1);
    wait_for(sub { find_event($waiter, 'queued') }, 'waiter queued');
    need(!find_event($waiter, 'admitted'),
         'queued-cancellation fixture was admitted before disconnect');
    close_request($waiter, 1);
    my $cancel_wait = wait_for(sub { find_event($waiter, 'cancel_wait') },
                                'queued request removal');
    record_check('queued cancellation has no admitted or executed work',
        $cancel_wait->{aborted} && $cancel_wait->{rc} == -4
        && !find_event($waiter, 'admitted') && !find_event($waiter, 'turn'), {
            cancel_event_seq => $cancel_wait->{event_seq},
            request_id => $waiter->{request_id},
        });
    need(!find_event($holder, 'done'),
         'queued cancellation was not observed while its slot remained occupied');
    close_request($holder, 1);
    wait_for(sub { find_event($holder, 'done') }, 'queued-cancellation holder cleanup');
    my $holder_done = done_event(request_events($holder));
    record_check('queued-cancellation holder released after disconnect',
        $holder_done->{aborted} && $holder_done->{count} < $holder_done->{total}, {
            first_partial_count => $held_partial->{count},
            final_count => $holder_done->{count},
        });
    my $after_waiter = ask(label => 'slot reuse after cancelled waiter',
        slot => 0, ids => \@reuse_ids, new => 6);
    equal_tokens('slot reuse after queued cancellation', $after_waiter, $baseline);
}

sub image_fixture {
    my ($which) = @_;
    my ($fh, $path) = tempfile("sequence-image-$which-XXXXXX",
        DIR => dirname($report_path), SUFFIX => '.bin', UNLINK => 0);
    binmode($fh);
    my @header = (1, 1, 1, $which == 0 ? 7 : 11, 2560);
    my @relative = $which == 0 ? (0, 2, 5) : (3, 1, 0);
    my @values = map {
        $which == 0 ? (($_ % 31) - 15) / 4 : ((($_ * 7) % 29) - 14) / 3
    } 0 .. 2559;
    my $bytes = pack('V5', @header) . pack('f<*', @values) . pack('V3', @relative);
    print {$fh} $bytes or die "cannot write image fixture: $!\n";
    close($fh) or die "cannot close image fixture: $!\n";
    push @{$report->{fixtures}}, {
        path => $path, sha256 => Digest::SHA::sha256_hex($bytes),
        header => \@header, relative_position => \@relative,
        embedding_formula => $which == 0
            ? '((j % 31) - 15) / 4, j=0..2559'
            : '(((j * 7) % 29) - 14) / 3, j=0..2559',
        format => 'little-endian u32 header; f32 embeddings; u32 positions',
    };
    return $path;
}

sub images_case {
    need($info->{images} >= 1, 'image isolation is required, not skippable');
    my $image_a = image_fixture(0);
    my $image_b = image_fixture(1);
    my @ids = (401 .. 408);
    my %plain = (ids => \@ids, new => 8);
    my $text = ask(%plain, slot => 0, label => 'image plain baseline');
    my $ia = ask(%plain, slot => 0, image => $image_a, label => 'image A serial baseline');
    my $ib = ask(%plain, slot => 1, image => $image_b, label => 'image B serial baseline');
    my %outputs = map { join(',', @{$_->{tokens}}) => 1 } ($text, $ia, $ib);
    record_check('image fixtures produce distinct observable model outputs',
        keys(%outputs) == 3, {
            text => $text->{label}, image_a => $ia->{label}, image_b => $ib->{label},
        });
    my ($with_image, $with_text) = pair_requests(
        {%plain, slot => 0, image => $image_a, label => 'interleaved image A'},
        {%plain, slot => 1, label => 'interleaved image-free text'});
    equal_tokens('image/text image parity', $with_image, $ia);
    equal_tokens('image/text text parity', $with_text, $text);
    my ($both_a, $both_b) = pair_requests(
        {%plain, slot => 0, image => $image_a, label => 'two-image A'},
        {%plain, slot => 1, image => $image_b, label => 'two-image B'});
    equal_tokens('two-image A parity', $both_a, $ia);
    equal_tokens('two-image B parity', $both_b, $ib);
    my ($swap_a, $swap_b) = pair_requests(
        {%plain, slot => 1, image => $image_a, label => 'swapped image A'},
        {%plain, slot => 0, image => $image_b, label => 'swapped image B'});
    equal_tokens('swapped image A parity', $swap_a, $ia);
    equal_tokens('swapped image B parity', $swap_b, $ib);
    for my $slot (0, 1) {
        my $r = ask(%plain, slot => $slot, label => "image-to-text reuse slot $slot");
        equal_tokens("image-to-text reuse slot $slot parity", $r, $text);
    }
    my $snap = ask(%plain, slot => 0, snap => 3,
                   label => 'text snapshot overlapping future image');
    equal_tokens('text snapshot baseline parity', $snap, $text);
    record_check('overlap regression created text snapshot at 3',
                 $snap->{done}{snapshot} == 3, {});
    my $overlap = ask(%plain, slot => 0, image => $image_a,
                      label => 'image starting before existing text snapshot');
    equal_tokens('image invalidates overlapping text snapshot output', $overlap, $ia);
    record_check('image does not resume past its first position',
                 $overlap->{done}{resumed} == 0, {
                     previous_snapshot => 3, image_start => 1,
                     resumed => $overlap->{done}{resumed},
                 });
    my $last_text = ask(%plain, slot => 0, label => 'plain request after overlap regression');
    equal_tokens('plain state restored after overlap regression', $last_text, $text);
}

my $ok = eval {
    need(-S $socket_path, 'SOCKET is not a Unix socket');
    need(-f $trace_path, 'TRACE_JSONL is not an existing regular file');
    open($trace_fh, '<', $trace_path) or die "cannot open trace $trace_path: $!\n";
    binmode($trace_fh);
    seek($trace_fh, 0, 2) or die "cannot seek trace: $!\n";
    $report->{trace_start_offset} = tell($trace_fh);
    my $raw_info = query_line('info');
    my @fields = split ' ', $raw_info;
    need(@fields >= 5 && $fields[0] eq 'info', 'invalid info response');
    $info = {
        positions => 0 + $fields[1], images => 0 + $fields[2],
        sequences => 0 + $fields[3], quantum => 0 + $fields[4],
    };
    my $raw_config = query_line('config');
    need($raw_config =~ /\Aconfig\s/, 'invalid config response');
    my %cfg;
    while ($raw_config =~ /(?:\A|\s)([A-Za-z_]+)=([^\s]+)/g) { $cfg{$1} = $2; }
    $config = \%cfg;
    $report->{info_raw} = $raw_info;
    $report->{config_raw} = $raw_config;
    $report->{config} = $config;
    $report->{info} = $info;
    for my $field (qw(rev dies layers maxt sequences quantum kv_host ple_disk
                      nblk cubin cubin_fnv img_rows)) {
        need(exists($cfg{$field}), "config lacks required field $field");
    }
    record_check('loaded revision matches explicit committed revision',
        $cfg{rev} eq $revision, { expected => $revision, loaded => $cfg{rev} });
    record_check('whole-model six-die configuration',
        $cfg{dies} == 6 && $cfg{layers} eq '48/48', {
            dies => $cfg{dies}, layers => $cfg{layers},
        });
    record_check('required concurrent capacity and image storage',
        $info->{sequences} >= 2 && $info->{positions} >= 128
        && $info->{images} >= 1, $info);
    record_check('info and config agree',
        $cfg{maxt} == $info->{positions}
        && $cfg{sequences} == $info->{sequences}
        && $cfg{quantum} == $info->{quantum}
        && $cfg{img_rows} == $info->{images}, {});
    need($cfg{cubin_fnv} =~ /\A[0-9a-f]{16}\z/
         && $cfg{cubin_fnv} ne '0000000000000000', 'loaded cubin has no usable identity');
    $capacity = $info->{positions};
    $quantum = $info->{quantum};
    need($quantum >= 1 && $quantum <= 128, 'invalid configured quantum');
    if (open(my $source, '<', $0)) {
        binmode($source);
        $report->{checker_sha256} = Digest::SHA->new(256)->addfile($source)->hexdigest;
        close($source);
    }
    if ($case eq 'all') {
        parity_and_snapshots();
        progress_case();
        cancellation_case();
        images_case();
    } elsif ($case eq 'progress') {
        progress_case();
    } else {
        cancellation_case();
    }
    tick();
    need(length($trace_buffer) == 0, 'trace ended with an incomplete JSON record');
    1;
};
my $error = $@;
# Only connections created by this checker are closed. No process is killed.
for my $r (@requests) { close_request($r, exists($r->{done}) ? 0 : 1); }
$report->{result} = $ok ? 'passed' : 'failed';
$report->{error} = $error unless $ok;
$report->{elapsed_seconds} = mono() - $started;
$report->{trace_events} = \@trace_events;
$report->{requests} = [map {
    my %copy = %$_;
    delete $copy{socket}; delete $copy{buffer};
    \%copy;
} @requests];
my $temporary = "$report_path.tmp.$$";
open(my $out, '>', $temporary) or die "cannot write report $temporary: $!\n";
print {$out} JSON::PP->new->canonical->pretty->encode($report)
    or die "cannot write report: $!\n";
close($out) or die "cannot close report: $!\n";
rename($temporary, $report_path) or die "cannot install report $report_path: $!\n";
close($trace_fh) if $trace_fh;
if ($ok) { print "sequence_check: passed case=$case report=$report_path\n"; exit 0; }
print STDERR "sequence_check: failed case=$case report=$report_path\n$error";
exit 1;
