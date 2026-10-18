#!/usr/bin/env perl
use strict; use warnings; use FindBin; use lib $FindBin::Bin;
use QsaEvidence qw(run write_text csv_rows);
use File::Temp 'tempdir'; use Test::More;
my ($out,$err,$rc)=run([$^X,'-e','print STDERR "e"x262144; print STDOUT "o"x262144; exit 7'],\%ENV,5);
is($rc,7,'nonzero child status retained');
is(length($out),262144,'stdout drained completely');
is(length($err),262144,'stderr drained concurrently');
($out,$err,$rc)=run([$^X,'-e','kill "TERM", $$; sleep 2'],\%ENV,5);
is($rc,-15,'signal termination retained');
my $dir=tempdir(CLEANUP=>1);
($out,$err,$rc)=run([$^X,'-e','print "before"; sleep 5'],\%ENV,.1,$dir);
is($rc,124,'wall timeout fails explicitly');
like($err,qr/RUNNER TIMEOUT/,'timeout included in raw evidence');
($out,$err,$rc)=run([$^X,'-MCwd=getcwd','-e','print getcwd'],\%ENV,5,$dir);
is($rc,0,'working-directory command succeeds');
is($out,$dir,'working directory applied only to child');
write_text("$dir/quoted.csv","a,b,c\n1,\"a,b\",\"first\nsecond\"\n2,\"a\"\"b\",3\n");
is_deeply(csv_rows("$dir/quoted.csv"),[
 {a=>'1',b=>'a,b',c=>"first\nsecond"},{a=>'2',b=>'a"b',c=>'3'}],
 'CSV preserves quoted comma, newline and escaped quote');
write_text("$dir/bad.csv","a,b\n1,2,3\n");
ok(!eval{csv_rows("$dir/bad.csv");1},'malformed CSV rejected');
done_testing();
