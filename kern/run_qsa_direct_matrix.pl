#!/usr/bin/env perl
use strict;
use warnings;
use FindBin;
use lib $FindBin::Bin;
use QsaEvidence qw(sha put write_text run absolute new_output utc);
use Getopt::Long 'GetOptions';
use JSON::PP;
use Time::HiRes qw(time sleep);
use Fcntl qw(:flock);
use File::Path 'make_path';
use File::Basename 'dirname';
$| = 1;

sub read_json {
 my ($path) = @_;
 open my $f, '<:raw', $path or die "open $path: $!\n";
 local $/;
 return JSON::PP->new->decode(<$f>);
}

sub checked_command {
 my ($cmd, $env, $path) = @_;
 my ($out, $err, $rc) = run($cmd, $env, 30);
 write_text($path, $out . $err);
 die "metadata command failed (exit $rc): @$cmd\n" if $rc != 0;
 return $out;
}

sub validate_build {
 my ($manifest, $root) = @_;
 my $b = read_json($manifest);
 ($b->{schema} // '') eq 'flash-sass.qsa-direct-build.v1'
  or die "unsupported build manifest\n";
 ($b->{source_revision} // '') =~ /^[0-9a-f]{40}$/
  or die "build manifest must identify a full source revision\n";
 defined($b->{build}{exit_code}) && $b->{build}{exit_code} == 0
  or die "build did not succeed\n";
 $b->{passed} && $b->{sources_unchanged}
  or die "build receipt is incomplete or its sources changed\n";
 ref($b->{sources_sha256}) eq 'HASH' && keys %{$b->{sources_sha256}}
  or die "build has no source fingerprints\n";
 for my $path (sort keys %{$b->{sources_sha256}}) {
  $path =~ m{^(?:kern|ref|maxas)/} && $path !~ m{(?:^|/)\.\.(?:/|$)}
   or die "invalid source path in manifest: $path\n";
  sha("$root/$path") eq $b->{sources_sha256}{$path}
   or die "source changed after build: $path\n";
 }
 for my $target (qw(binary cubin)) {
  ref($b->{$target}) eq 'HASH' or die "missing build target $target\n";
  sha($b->{$target}{path}) eq $b->{$target}{sha256}
   or die "$target changed after build\n";
 }
 for my $path ('kern/die.sass', 'kern/dieops.h') {
  sha("$root/$path") eq ($b->{generated_sources_sha256}{$path} // '')
   or die "generated source changed after build: $path\n";
 }
 return $b;
}

sub preparing_gpu_workers {
 my ($snapshot, $gpu, $index) = @_;
 my @busy;
 for my $line (split /\n/, $snapshot) {
  next unless $line =~ /^\s*(\d+)\s+\d+\s+\d+\s+\S+\s+(\S+)\s*$/;
  my ($pid, $comm) = ($1, $2);
  next unless $comm =~ /^(?:(?:qsa|model|attn|gemv|idx|die).*_test|memtest|ncu|nvprof|compute-sanitiz)$/;
  if (!open(my $envfile, '<:raw', "/proc/$pid/environ")) {
   push @busy, $pid unless $!{ENOENT};
   next;
  } else {
   my $process_env;
   { local $/; $process_env = <$envfile>; }
   close $envfile;
   my ($visible) = map { /^CUDA_VISIBLE_DEVICES=(.*)$/s ? $1 : () }
                       split /\0/, ($process_env // '');
   if (!defined $visible) { push @busy, $pid; next; }
   for my $device (split /,/, $visible) {
    $device =~ s/^\s+|\s+$//g;
    if ($device eq "$index" || $device eq $gpu ||
        ($device =~ /^GPU-/ && index($gpu, $device) == 0)) {
     push @busy, $pid;
     last;
    }
   }
  }
 }
 return @busy;
}

# Hold the same per-UUID lock used by the other workers, and check both
# NVIDIA's process inventory and ps before every child gets a CUDA context.
# A busy GPU is waited for; this runner never signals another worker.
sub wait_idle {
 my ($gpu, $env, $out, $index) = @_;
 my $attempt = 0;
 while (1) {
  my $stem = sprintf('%s/preflight/%03d-%03d', $out, $index, $attempt++);
  my $query = checked_command(
   ['nvidia-smi', "--id=$gpu",
    '--query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.mem,temperature.gpu,memory.used,utilization.gpu',
    '--format=csv'], $env, "$stem-gpu.csv");
  my $apps = checked_command(
   ['nvidia-smi', "--id=$gpu",
    '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
    '--format=csv,noheader,nounits'], $env, "$stem-compute.csv");
  my $processes = checked_command(['ps', '-eo', 'pid,ppid,etimes,stat,comm'],
                                 $env, "$stem-ps.txt");
  my @lines = grep { /\S/ } split /\n/, $query;
  @lines == 2 or die "GPU query must return exactly one device\n";
  my @fields = split /,\s*/, $lines[1];
  @fields == 9 && $fields[1] eq $gpu && $fields[2] eq 'Tesla M60'
   or die "GPU identity is not the selected Tesla M60\n";
  my ($util) = $fields[8] =~ /^(\d+)\s*%?$/;
  defined($util) or die "unreadable GPU utilization\n";
  my @preparing = preparing_gpu_workers($processes, $gpu, $fields[0]);
  if ($apps !~ /\S/ && $util == 0 && !@preparing) {
   return {gpu_csv => "$stem-gpu.csv", compute_csv => "$stem-compute.csv",
           ps => "$stem-ps.txt", attempts => $attempt};
  }
  print "GPU $gpu is busy; waiting (preflight $index, attempt $attempt, preparing PIDs @preparing)\n";
  sleep 5;
 }
}

sub main {
 my ($binary, $out, $quick, $timeout, $manifest, $lock) =
    (undef, undef, 0, 180, undef, undef);
 GetOptions('binary=s' => \$binary, 'out=s' => \$out, 'quick' => \$quick,
            'timeout=i' => \$timeout, 'build-manifest=s' => \$manifest,
            'gpu-lock=s' => \$lock)
  or die "invalid options\n";
 defined($out) && defined($manifest) && defined($lock) && !@ARGV
  or die "usage: $0 --out NEW_DIRECTORY --build-manifest FILE --gpu-lock FILE [--quick] [--timeout SECONDS]\n";
 $timeout > 0 or die "--timeout must be positive\n";
 my $gpu = $ENV{CUDA_VISIBLE_DEVICES} // '';
 $gpu =~ /^GPU-[0-9a-fA-F-]+$/
  or die "CUDA_VISIBLE_DEVICES must name exactly one full GPU UUID\n";
 my $root = absolute("$FindBin::Bin/..");
 $manifest = absolute($manifest);
 my $build = validate_build($manifest, $root);
 $binary = absolute($binary // $build->{binary}{path});
 sha($binary) eq $build->{binary}{sha256}
  or die "--binary does not match the verified build\n";
 my $cubin = absolute($build->{cubin}{path});
 if (defined $ENV{DIE_CUBIN}) {
  absolute($ENV{DIE_CUBIN}) eq $cubin
   or die "DIE_CUBIN does not match the verified build\n";
 }
 -d dirname($lock) or die "GPU lock directory does not exist\n";
 open my $reservation, '>>', $lock or die "open GPU lock $lock: $!\n";
 while (!flock($reservation, LOCK_EX | LOCK_NB)) {
  $!{EWOULDBLOCK} || $!{EAGAIN} or die "flock $lock: $!\n";
  print "Waiting for GPU reservation $lock\n";
  sleep 5;
 }
 $out = new_output($out);
 make_path("$out/preflight");
 my %env = %ENV;
 delete $env{$_} for grep { /^QSA_/ } keys %env;
 $env{DIE_CUBIN} = $cubin;
 my @ranges = $quick ? ([1,4], [2047,4], [2052,4], [3100,4])
   : ([1,4], [127,8], [255,8], [511,8], [767,8], [1023,8],
      [1279,8], [1535,8], [1791,8], [2047,4], [2051,4], [2055,4],
      [3100,4], [8000,4], [16383,9], [65000,4], [262000,4], [262141,4]);
 my @cases;
 for my $host (0,1) {
  for my $range (@ranges) {
   push @cases, [$range->[0],512,$range->[1],4242,$host,'hash',0,undef];
  }
  push @cases, [65,8,4,4242,$host,'hash',1,undef];
  for my $pattern (qw(oldest newest clustered scattered equal adjacent)) {
   push @cases, [3101,512,4,4242,$host,$pattern,0,undef];
  }
 }
 if (!$quick) {
  for my $seed (0,77) {
   for my $visible (2052,8000,65000) {
    push @cases, [$visible,512,4,$seed,0,'hash',0,undef];
   }
  }
 }
 for my $injection (qw(selection gather output)) {
  push @cases, [2052,512,1,4242,0,'hash',0,$injection];
 }
 my $info = {
  schema => 'flash-sass.qsa-direct-matrix.v2', started_utc => utc(),
  source_revision => $build->{source_revision},
  build_manifest_sha256 => sha($manifest),
  sources_sha256 => $build->{sources_sha256},
  binary => $binary, binary_sha256 => sha($binary),
  cubin => $cubin, cubin_sha256 => sha($cubin),
  CUDA_VISIBLE_DEVICES => $gpu, gpu_lock => absolute($lock),
  scope => 'synthetic-history component parity; no whole-model quality or performance claim',
  quick => $quick ? JSON::PP::true : JSON::PP::false,
  cases => [], passed => JSON::PP::false
 };
 write_text("$out/build.json", JSON::PP->new->canonical->pretty->encode($build));
 put("$out/results.json", $info);
 for my $i (0..$#cases) {
  my ($visible,$budget,$steps,$seed,$host,$pattern,$replay,$injection) = @{$cases[$i]};
  my $tag = sprintf('%03d-n%d-b%d-s%d-h%d-%s',$i,$visible,$budget,$seed,$host,$pattern);
  $tag .= '-replay' if $replay;
  $tag .= '-inject-'.$injection if defined $injection;
  my $preflight = wait_idle($gpu, \%env, $out, $i);
  sha($manifest) eq $info->{build_manifest_sha256}
   or die "build manifest changed while waiting\n";
  validate_build($manifest, $root);
  sha($binary) eq $info->{binary_sha256}
   or die "binary changed while waiting\n";
  my %runenv = (%env, QSA_DIRECT_SEED => "$seed", QSA_HOSTKV => "$host",
               QSA_PATTERN => $pattern, QSA_REPLAY_PREFIX => "$replay");
  $runenv{QSA_DIRECT_INJECT} = $injection if defined $injection;
  my @cmd = ($binary, "$visible", "$budget", "$steps");
  my $started = time;
  my ($stdout,$stderr,$rc) = run(\@cmd, \%runenv, $timeout);
  my $log = "$out/$tag.log";
  write_text($log, $stdout.$stderr);
  my @rows;
  while ($stdout =~ /^visible=(\d+) complete=(\d+) tail=(\d+) selected=(\d+) keys=(\d+) guarded_allocations=(\d+) exclusion=(yes|no) (EXACT|FAIL)$/mg) {
   push @rows, [$1,$2,$3,$4,$5,$6,$7,$8];
  }
  my $expected = defined($injection) ? 3 : 0;
  my $passed = $rc == $expected && @rows == $steps;
  my @source = $stdout =~ /^direct QSA: SYNTHETIC HISTORY source=(\S+)/mg;
  $passed &&= @source == 1 && $source[0] eq $build->{source_revision};
  for my $j (0..$#rows) {
   my ($n,$complete,$tail,$selected,$keys,$guards,$excluded,$status) = @{$rows[$j]};
   my $wantselected = int($n/4) < $budget ? int($n/4) : $budget;
   $passed &&= $n == $visible+$j && $complete == int($n/4) &&
     $tail == $n%4 && $selected == $wantselected && $keys == 4*$selected+$tail &&
     $guards > 0 && (($excluded eq 'yes') == (int($n/4)>$budget)) &&
     $status eq (defined($injection) ? 'FAIL' : 'EXACT');
  }
  my $cutoff;
  if ($pattern eq 'equal' || $pattern eq 'adjacent') {
   if ($stdout =~ /^cutoff pattern=(\w+) visible=(\d+) budget=(\d+) selected_id=(\d+) excluded_id=(\d+) selected_bits=([0-9a-f]{8}) excluded_bits=([0-9a-f]{8}) relation=(\w+) EXACT$/m) {
    $cutoff = {pattern=>$1,visible=>0+$2,budget=>0+$3,selected_id=>0+$4,
               excluded_id=>0+$5,selected_bits=>$6,excluded_bits=>$7,relation=>$8};
    $passed &&= $1 eq $pattern && $2 == $visible && $3 == $budget &&
      $8 eq $pattern && (($pattern eq 'equal' && hex($6)==hex($7) && $4<$5) ||
                         ($pattern eq 'adjacent' && hex($6)==hex($7)+1));
   } else { $passed = 0; }
  }
  $passed &&= $stdout =~ /^prefix history=64 kv=EXACT ring=EXACT pool=EXACT$/m if $replay;
  my %qenv = map { $_ => $runenv{$_} } grep { /^QSA_/ } keys %runenv;
  push @{$info->{cases}}, {
   case=>$tag,command=>\@cmd,environment=>\%qenv,preflight=>$preflight,
   exit_code=>$rc,expected_exit_code=>$expected,seconds=>time-$started,
   parsed_positions=>scalar(@rows),positions=>\@rows,cutoff=>$cutoff,
   log_sha256=>sha($log),passed=>$passed ? JSON::PP::true : JSON::PP::false
  };
  put("$out/results.json",$info);
  print "$tag ", ($passed ? 'PASS' : 'FAIL'), "\n";
  return 1 unless $passed;
 }
 $info->{passed} = JSON::PP::true;
 $info->{finished_utc} = utc();
 put("$out/results.json",$info);
 return 0;
}
my $rc = eval { main() };
if ($@) { print STDERR $@; exit 1; }
exit $rc;
