#!/usr/bin/perl
# Identical instructions/shared memory; raise declared registers for occupancy control.
use strict;
use warnings;
use MaxAs::Cubin;
use Digest::SHA 'sha256_hex';
@ARGV==2 or die "usage: pin_q3_regs.pl INPUT OUTPUT\n";
my ($input,$output)=@ARGV;
my $c=MaxAs::Cubin->new($input);
my $k=$c->getKernel('gemv_q3_k') or die 'missing Q3 entry';
my $hash=sha256_hex(pack 'Q*',@{$k->{KernelData}});
$c->modifyKernel(Kernel=>$k,RegCnt=>249,BarCnt=>$k->{BarCnt},
    KernelData=>$k->{KernelData},ExitOffsets=>$k->{ParamSec}{ExitOffsets},
    CTAIDOffsets=>$k->{ParamSec}{CTAIDOffsets},CTAIDZUsed=>$k->{ParamSec}{CTAIDZUsed});
$c->write($output);
my $copy=MaxAs::Cubin->new($output)->getKernel('gemv_q3_k');
sha256_hex(pack 'Q*',@{$copy->{KernelData}}) eq $hash or die 'instruction bytes changed';
print "register control: identical instruction_sha256=$hash registers=249\n";
