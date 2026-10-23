#!/usr/bin/env perl
use strict;
use warnings;
use FindBin;
use lib $FindBin::Bin;
use QsaEvidence qw(sha put write_text run absolute new_output utc);
use Getopt::Long qw(GetOptions);
use Digest::SHA qw(sha256_hex);
use JSON::PP;
use Sys::Hostname qw(hostname);
use Time::HiRes qw(time);
use File::Find qw(find);
$| = 1;

sub source_path {
 my ($p) = @_;
 return 1 if $p =~ m{\Amaxas-patch/};
 return 0 unless $p =~ m{\A(?:kern|ref|maxas)/};
 return 0 if $p eq 'kern/die.sass' || $p eq 'kern/dieops.h';
 return 1 if $p =~ m{(?:\A|/)(?:Makefile|GNUmakefile|makefile)\z};
 return $p =~ /\.(?:c|h|cu|cuh|ptx|sass|s|S|asm|inc|pl|pm|t|sh|bash|mk|py|rb|rs|zig|go|lua)\z/;
}
sub digest_sources {
 my ($sources) = @_;
 return sha256_hex(join '', map { "$_\0$sources->{$_}\0" } sort keys %$sources);
}
sub main {
 my ($out, $cuda, $cc, $ptxas11, $timeout) =
    (undef, '/opt/cuda', '/usr/bin/cc', '/opt/cuda-11.4/bin/ptxas', 1800);
 GetOptions('out=s'=>\$out, 'cuda=s'=>\$cuda, 'cc=s'=>\$cc,
            'ptxas11=s'=>\$ptxas11, 'timeout=i'=>\$timeout)
  or die "invalid options\n";
 defined($out) && !@ARGV && $timeout > 0
  or die "usage: $0 --out NEW_DIRECTORY [--cuda DIR] [--cc PATH] [--ptxas11 PATH] [--timeout SECONDS]\n";
 my $root = absolute("$FindBin::Bin/..");
 $cuda = absolute($cuda); $cc = absolute($cc); $ptxas11 = absolute($ptxas11);
 -x $cc or die "compiler is not executable: $cc\n";
 -x $ptxas11 or die "ptxas is not executable: $ptxas11\n";

 # Experimental generator flags and compiler overrides are not inherited.
 my %env = (PATH=>"$cuda/bin:/usr/local/bin:/usr/bin:/bin",
            LANG=>'C', LC_ALL=>'C', TZ=>'UTC');
 $env{HOME} = $ENV{HOME} if defined $ENV{HOME};
 my $git = sub {
  my (@args) = @_;
  my ($stdout,$stderr,$rc) = run(['git',@args], \%env, 30, $root);
  $rc == 0 or die "git @args failed ($rc): $stderr\n";
  return $stdout;
 };
 my $top = $git->('rev-parse','--show-toplevel'); $top =~ s/\s+\z//;
 absolute($top) eq $root or die "wrapper is not at repository root\n";
 my $revision = $git->('rev-parse','--verify','HEAD'); $revision =~ s/\s+\z//;
 $revision =~ /\A[0-9a-f]{40}\z/ or die "invalid source revision\n";
 my @sources = sort grep { length($_) && source_path($_) }
    split /\0/, $git->('ls-files','-z','--','kern','ref','maxas-patch');
 @sources or die "no tracked build sources found\n";
 grep { $_ eq 'kern/build_qsa_direct.pl' } @sources
  or die "commit the build wrapper before producing acceptance evidence\n";
 my @untracked = grep { length($_) && source_path($_) }
    split /\0/, $git->('ls-files','--others','--exclude-standard','-z',
                      '--','kern','ref','maxas-patch');
 @untracked and die "untracked source files: @untracked\n";
 my (undef,$dirty_error,$dirty_rc) =
    run(['git','diff','--quiet','HEAD','--',@sources], \%env, 30, $root);
 $dirty_rc == 0
  or die "tracked build sources are dirty or could not be checked (exit $dirty_rc): $dirty_error\n";
 -f "$root/maxas/bin/maxas.pl"
  or die "MaxAs is missing: install the checkout and apply maxas-patch/ as described in README.md\n";
 my @external;
 find({no_chdir=>1,wanted=>sub {
  return unless -f $File::Find::name && $File::Find::name =~ /\.(?:pl|pm)\z/;
  push @external, substr($File::Find::name,length($root)+1);
 }}, "$root/maxas/bin", "$root/maxas/lib");
 @external or die "MaxAs source inventory is empty\n";
 push @sources, sort @external;
 my %hashes = map { $_=>sha("$root/$_") } @sources;
 my $source_digest = digest_sources(\%hashes);
 $out = new_output($out);
 my @command = ('make','-B','-C',"$root/kern","CC=$cc","CUDA=$cuda",
                "FLASH_REV=$revision","PTXAS11=$ptxas11",
                'qsa_direct_test','die.cubin');
 my $info = {
  schema=>'flash-sass.qsa-direct-build.v1',started_utc=>utc(),
  hostname=>hostname(),source_root=>$root,source_revision=>$revision,
  sources_sha256=>\%hashes,sources_digest_sha256=>$source_digest,
  environment=>\%env,
  toolchain=>{compiler_path=>$cc,compiler_sha256=>sha($cc),cuda_root=>$cuda,
              ptxas11_path=>$ptxas11,ptxas11_sha256=>sha($ptxas11)},
  binary=>{path=>"$root/kern/qsa_direct_test",sha256=>undef},
  cubin=>{path=>"$root/kern/die.cubin",sha256=>undef},
  build=>{command=>\@command,exit_code=>undef,
           stdout=>"$out/build.stdout.log",stderr=>"$out/build.stderr.log"},
  passed=>JSON::PP::false
 };
 put("$out/build.json",$info);
 my ($maxas_head,$maxas_head_err,$maxas_head_rc) =
    run(['git','-C',"$root/maxas",'rev-parse','HEAD'],\%env,30,$root);
 my ($maxas_patch,$maxas_patch_err,$maxas_patch_rc) =
    run(['git','-C',"$root/maxas",'diff','--binary','HEAD','--','bin','lib'],\%env,30,$root);
 $maxas_head_rc == 0 && $maxas_patch_rc == 0
  or die "MaxAs checkout provenance unavailable: $maxas_head_err$maxas_patch_err\n";
 $maxas_head =~ s/\s+\z//;
 write_text("$out/maxas.patch",$maxas_patch);
 $info->{external_maxas} = {upstream=>'https://github.com/NervanaSystems/maxas',
   revision=>$maxas_head,source_paths=>\@external,
   patch=>"$out/maxas.patch",patch_sha256=>sha("$out/maxas.patch")};
 my @versions = (['compiler',[$cc,'--version']],
                 ['make',['make','--version']],
                 ['ptxas11',[$ptxas11,'--version']]);
 push @versions,['nvcc',["$cuda/bin/nvcc",'--version']] if -x "$cuda/bin/nvcc";
 for my $query (@versions) {
  my ($name,$cmd) = @$query;
  my ($stdout,$stderr,$rc) = run($cmd,\%env,30,$root);
  $info->{toolchain}{$name} = {command=>$cmd,stdout=>$stdout,stderr=>$stderr,exit_code=>$rc};
  $rc == 0 or die "toolchain query $name failed with exit $rc\n";
 }
 put("$out/build.json",$info);
 my $start = time;
 my ($stdout,$stderr,$rc) = run(\@command,\%env,$timeout,$root);
 write_text("$out/build.stdout.log",$stdout);
 write_text("$out/build.stderr.log",$stderr);
 $info->{build}{exit_code}=$rc; $info->{build}{seconds}=time-$start;
 $info->{finished_utc}=utc(); put("$out/build.json",$info);
 $rc == 0 or die "build failed with exit $rc; see $out/build.json\n";
 my %after = map { $_=>sha("$root/$_") } @sources;
 my $unchanged = digest_sources(\%after) eq $source_digest;
 $info->{sources_unchanged}=$unchanged?JSON::PP::true:JSON::PP::false;
 put("$out/build.json",$info);
 $unchanged or die "source inputs changed during the build\n";
 for my $kind (qw(binary cubin)) {
  my $path=$info->{$kind}{path};
  -f $path && -s $path or die "build did not produce a nonempty $kind: $path\n";
  $info->{$kind}{sha256}=sha($path); $info->{$kind}{bytes}=-s $path;
 }
 my %generated;
 for my $path ('kern/die.sass','kern/dieops.h') {
  $generated{$path}=sha("$root/$path");
 }
 $info->{generated_sources_sha256}=\%generated;
 $info->{passed}=JSON::PP::true;
 put("$out/build.json",$info);
 print "$out/build.json\n";
 return 0;
}
my $rc = eval { main() };
if ($@) { print STDERR $@; exit 1; }
exit $rc;
