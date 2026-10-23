#!/usr/bin/env perl
use strict;
use warnings;
use FindBin;
use JSON::PP;
use List::Util qw(min max sum);
my $root = $ARGV[0] // $FindBin::Bin;
sub read_json {
    my ($path) = @_;
    open my $f, '<:raw', $path or die "Read $path: $!";
    local $/; my $text = <$f>;
    close $f or die "Close $path: $!";
    return decode_json($text);
}
sub stats {
    my @values = @_;
    return {count=>0} unless @values;
    return {count=>scalar(@values), min=>min(@values), max=>max(@values), mean=>sum(@values)/@values};
}
my $run = read_json("$root/evidence/results.json");
die "Resident validation did not pass\n" unless $run->{status} eq 'PASS' && $run->{restoration}{status} eq 'PASS';
my $summary = {status=>'PASS', source_sha=>$run->{source_sha},
    started_utc=>$run->{started_utc}, finished_utc=>$run->{finished_utc},
    candidate_configs=>$run->{candidate_configs}, cases=>[], overhead_ms=>{},
    scope=>'One instrumented acceptance run, not a throughput benchmark. Host CLOCK_MONOTONIC intervals only; server write return is not client receipt. Device and host epochs are never subtracted.'};
my %samples;
for my $pair (0,1) {
    for my $interface ('cli','http') {
        my $data = read_json("$root/evidence/$interface-pair$pair/results.json");
        die "Missing successful $interface-pair$pair results\n" unless $data->{status} eq 'PASS';
        for my $case (@{$data->{cases}}) {
            if ($case->{status} eq 'SKIP') {
                die "Unexpected skipped case\n" unless $interface eq 'cli' && $pair == 0
                    && $case->{case} eq 'paired_prompt';
                push @{$summary->{cases}}, {two_position=>$pair, interface=>$interface,
                    case=>$case->{case}, status=>'SKIP', reason=>$case->{reason}};
                next;
            }
            die "Failed case\n" unless $case->{status} eq 'PASS';
            my $t = $interface eq 'cli' ? $case->{timings} : $case->{timings}{engine};
            my $row = {two_position=>$pair, interface=>$interface, case=>$case->{case}, status=>'PASS',
                map {$_=>$t->{$_}} qw(generated_tokens decode_intervals resumed_positions positions_executed
                prompt_positions_processed build_ms run_ms first_observed_ms post_first_ms
                device_decode_ms device_decode_tokens_per_second decode_tokens_per_second)};
            if ($interface eq 'http') {
                $row->{socket_decode_tokens_per_second} = $case->{timings}{http}{socket_decode_tokens_per_second};
                $row->{overhead_ms} = $case->{overhead};
                for my $key (qw(first_observation_to_text_send_ms record_construction_to_response_write_return_ms)) {
                    push @{$samples{$key}}, $case->{overhead}{$key} if defined $case->{overhead}{$key};
                }
            }
            my $events = $interface eq 'cli' ? ($case->{host}{events}//[]) : ($case->{token_events}//[]);
            for my $event (@$events) {
                for my $key (qw(callback_ms observation_to_callback_return_ms observation_to_socket_parse_ms)) {
                    push @{$samples{$key}}, $event->{$key} if defined $event->{$key};
                }
            }
            push @{$summary->{cases}}, $row;
        }
    }
}
$summary->{overhead_ms}{$_} = stats(@{$samples{$_}}) for sort keys %samples;
print JSON::PP->new->canonical->pretty->encode($summary);
