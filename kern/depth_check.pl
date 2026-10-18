#!/usr/bin/env perl
# Exit 0 correct, 1 wrong answer, 2 invalid protocol/timings.
use strict;use warnings;use JSON::PP;use B qw(svref_2object SVp_IOK SVp_NOK);use POSIX qw(isfinite);
sub number{my($v)=@_;return 0 if!defined($v)||ref($v);my$f=svref_2object(\$v)->FLAGS;return($f&(SVp_IOK|SVp_NOK))&&isfinite($v);}
sub count{my($v,$name)=@_;number($v)&&$v>=0&&int($v)==$v or die"invalid timing count: $name\n";return$v;}
sub nonnegative{my($v,$name)=@_;number($v)&&$v>=0 or die"invalid timing value: $name\n";return$v;}
sub boolean{my($v,$expected)=@_;return ref($v) eq'JSON::PP::Boolean'&&!!$v==!!$expected;}
sub rate_matches {
    my ($rate, $n, $ms, $phase) = @_;
    $phase //= 'decode';
    defined($rate) or die "unknown $phase rate with measured work\n";
    nonnegative($rate, "${phase}_per_second");
    # flashd independently rounds milliseconds to 0.1 and rates to 0.01.
    # Their nonnegative rounding intervals must contain the same work count.
    # This also handles positive durations or rates that serialize as zero.
    # The extra 1e-6 margins cover binary noise at decimal rounding boundaries.
    my $min_ms = $ms > .050001 ? $ms - .050001 : 0;
    my $min_rate = $rate > .005001 ? $rate - .005001 : 0;
    my $work = 1000 * $n;
    my $units = $phase eq 'prefill' ? 'uncached prompt tokens' : 'intervals';
    $work >= $min_ms * $min_rate
        && $work <= ($ms + .050001) * ($rate + .005001)
        or die "$phase rate does not match $units / milliseconds\n";
}
sub prompt_rate_matches {
    my ($t) = @_;
    my $cached = exists($t->{cache_n}) ? count($t->{cache_n}, 'cache_n') : 0;
    $cached <= $t->{prompt_n} or die "cached prompt count exceeds prompt count\n";
    my $processed = $t->{prompt_n} - $cached;
    my ($rate, $ms) = @{$t}{qw(prompt_per_second prompt_ms)};
    nonnegative($rate, 'prompt_per_second') if defined($rate);
    if (!$processed) {
        !defined($rate) || $rate == 0
            or die "a fully cached prompt cannot report positive prefill throughput\n";
        return;
    }
    # Legacy zero-duration records have no measurable prefill rate. Canonical
    # records may express that absence with null. Neither advertises throughput.
    return if $ms == 0 && (!defined($rate) || $rate == 0);
    defined($rate) or die "unknown prompt rate with measured prompt time\n";
    rate_matches($rate, $processed, $ms, 'prefill');
}

sub validate{my($response)=@_;ref($response) eq'HASH'&&!exists($response->{error}) or die"API error or non-object response\n";my$choices=$response->{choices};ref($choices) eq'ARRAY'&&@$choices&&ref($choices->[0]) eq'HASH'&&ref($choices->[0]{message}) eq'HASH' or die"missing answer or timings\n";my$answer=$choices->[0]{message}{content};defined($answer)&&!ref($answer)&&$answer=~/\S/&&!(svref_2object(\$answer)->FLAGS&(SVp_IOK|SVp_NOK)) or die"missing answer text\n";my$t=$response->{timings};ref($t) eq'HASH' or die"missing timings\n";for(qw(prompt_n predicted_n decode_intervals)){count($t->{$_},$_);}$t->{prompt_n}>0&&$t->{predicted_n}>0 or die"request did not process a prompt and generate an answer\n";$t->{decode_intervals}==$t->{predicted_n}-1 or die"decode interval count differs from generated outputs minus one\n";for(qw(prompt_ms predicted_ms decode_ms)){nonnegative($t->{$_},$_);}abs($t->{predicted_ms}-$t->{decode_ms})<=.101 or die"predicted/decode milliseconds differ\n";prompt_rate_matches($t);if(defined($t->{predicted_per_second})){nonnegative($t->{predicted_per_second},'predicted_per_second');}if($t->{decode_intervals}){rate_matches($t->{predicted_per_second},$t->{decode_intervals},$t->{decode_ms});}else{!defined($t->{predicted_per_second})||$t->{predicted_per_second}==0 or die"a single output cannot measure a decode rate\n";}
for my $section (qw(engine http)) {
    !exists($t->{$section}) || ref($t->{$section}) eq 'HASH'
        or die "invalid $section timing section: expected an object\n";
}
my$e=ref($t->{engine}) eq'HASH'?$t->{engine}:exists($t->{counts_known})?$t:undef;if($e){boolean($e->{counts_known},1)&&boolean($e->{counts_final},1)&&boolean($e->{aborted},0) or die"canonical timing counts unknown, unfinished or aborted\n";count($e->{error_code},'error_code')==0 or die"canonical engine error\n";count($e->{generated_tokens},'generated_tokens')==$t->{predicted_n} or die"canonical generated count differs\n";count($e->{decode_intervals},'engine.decode_intervals')==$t->{decode_intervals} or die"canonical decode count differs\n";count($e->{prompt_tokens},'prompt_tokens')==$t->{prompt_n} or die"canonical prompt count differs\n";}
if(ref($t->{http}) eq'HASH'){my$h=$t->{http};(!exists($h->{aborted})||boolean($h->{aborted},0))&&(!exists($h->{complete})||boolean($h->{complete},1)) or die"HTTP generation incomplete or aborted\n";}return($answer,$t);}
sub main{@ARGV==3 or die"usage: depth_check.pl response.json expected-code paragraphs\n";my($path,$code,$paragraphs)=@ARGV;$code=~/\A[0-9a-f]{6}\z/&&$paragraphs=~/\A[0-9]+\z/&&$paragraphs>0 or die"invalid expected code or paragraph count\n";open my$f,'<:raw',$path or die"$path: $!";local$/;my$response=decode_json(<$f>);my($answer,$t)=validate($response);my$passed=index($answer,$code)>=0;my$prefill=defined($t->{prompt_per_second})?$t->{prompt_per_second}:'null';my$decode=defined($t->{predicted_per_second})?$t->{predicted_per_second}:'null';printf"%s paragraphs  prompt %s tok  prefill %s tok/s  decode %s tok/s (%s intervals in %.1f ms)  %s\n",$paragraphs,$t->{prompt_n},$prefill,$decode,$t->{decode_intervals},$t->{decode_ms},$passed?'PASS':'FAIL';return$passed?0:1;}
my$rc;eval{$rc=main();1} or do{my$e=$@||'unknown error';print STDERR"depth: invalid response: $e";exit 2;};exit$rc;
