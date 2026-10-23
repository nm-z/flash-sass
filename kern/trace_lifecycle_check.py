#!/usr/bin/env python3
"""Real resident-core trace lifecycle regression; no mocked GPU evidence."""
import argparse
import gzip
import hashlib
import json
import pathlib
import socket
import uuid


def fields(line):
    return dict(x.split('=', 1) for x in line.split()[1:] if '=' in x)


def need(test, message):
    if not test:
        raise AssertionError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class Core:
    def __init__(self, path, timeout):
        self.s = socket.socket(socket.AF_UNIX)
        self.s.settimeout(timeout)
        self.s.connect(path)
        self.f = self.s.makefile('r')

    def send(self, command):
        self.s.sendall((command + '\n').encode())

    def line(self):
        line = self.f.readline()
        if not line:
            raise RuntimeError('core disconnected')
        return line.rstrip('\n')

    def close(self, abort=False):
        if abort:
            try:
                self.s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.f.close()
        self.s.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--socket', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--source-sha', required=True)
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--pair-check', action='store_true',
                   help='also test paired prefill/resume when config reports two_position=1')
    a = p.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=False)

    class Log(list):
        def append(self, line):
            super().append(line)
            with (out / 'protocol.txt').open('a') as f:
                f.write(line + '\n')

    protocol = Log()
    results = []
    last_request = -1

    def checkpoint(status='RUNNING'):
        (out / 'results.json').write_text(
            json.dumps(dict(status=status, cases=results), indent=2) + '\n')

    def connect():
        protocol.append('< connecting ' + a.socket + ' >')
        return Core(a.socket, a.timeout)

    def one(c, command):
        protocol.append('> ' + command)
        c.send(command)
        reply = c.line()
        protocol.append(reply)
        return reply

    def run(c, new, ids):
        command = 'chat ' + str(new) + ' ' + ' '.join(map(str, ids))
        protocol.append('> ' + command)
        c.send(command)
        tokens = []
        lines = []
        while True:
            line = c.line()
            protocol.append(line)
            lines.append(line)
            if line.startswith('tok '):
                tokens.append(int(line.split()[1]))
            elif line.startswith('done ') or line.startswith('error:'):
                return tokens, line, lines
            else:
                raise AssertionError('unexpected response ' + line)

    def capture(c, case, kind, die=None):
        command = kind if die is None else f'{kind} {die}'
        protocol.append('> ' + command)
        c.send(command)
        first = c.line()
        protocol.append(first)
        path = out / (case + '-' + kind + ('' if die is None else f'-{die}') + '.txt.gz')
        with gzip.open(path, 'wt') as f:
            f.write(first + '\n')
            if not first.startswith('error:'):
                need(first.startswith(kind + '_begin '), 'missing trace header: ' + first)
                while True:
                    line = c.line()
                    f.write(line + '\n')
                    if line == kind + '_end':
                        protocol.append(line)
                        break
        return first, fields(first), str(path.name)

    def statehash(c, case):
        path = out / (case + '-statehash.txt')
        protocol.append('> statehash')
        c.send('statehash')
        with path.open('w') as f:
            first = c.line()
            protocol.append(first)
            f.write(first + '\n')
            need(first.startswith('statehash positions='), 'missing statehash header: ' + first)
            meta = fields(first)
            while True:
                line = c.line()
                protocol.append(line)
                f.write(line + '\n')
                if line == 'statehash done':
                    break
                need(line.startswith('statehash '), 'unexpected statehash response: ' + line)
        need(meta.get('positions') == '7', 'paired case processed the wrong number of positions')
        need(meta.get('paired_positions') == '2', 'paired case did not execute two paired positions')
        return dict(header=first, raw_file=path.name, positions=7, paired_positions=2)

    def collect(c, case, status, aborted, ended, requested, completed=None,
                error=False, first_position=1, statehash_record=None):
        nonlocal last_request
        headers = []
        files = []
        common = None
        for kind, die in [('trace', d) for d in range(dies)] + [('hosttrace', None)]:
            line, m, path = capture(c, case, kind, die)
            files.append(path)
            headers.append(line)
            need(line.startswith('error:') == error, 'wrong trace availability: ' + line)
            need(m.get('status') == status and m.get('aborted') == str(aborted), 'wrong status: ' + line)
            need(m.get('ended') == str(ended), 'wrong end flag: ' + line)
            need(int(m['requested']) == requested, 'wrong requested positions: ' + line)
            if completed is not None:
                need(int(m['completed']) == completed, 'wrong completed positions: ' + line)
            else:
                need(first_position + 1 <= int(m['completed']) < requested,
                     'cancellation did not interrupt the request')
            shared = {k: m[k] for k in ('request', 'status', 'aborted', 'ended', 'completed', 'requested')}
            if common is None:
                common = shared
            else:
                need(shared == common, 'different lifecycle metadata across traces')
            if not error:
                need(int(m['first']) == first_position, 'wrong traced first position')
                need(int(m['count']) == min(int(m['completed']) - first_position, trace_cap),
                     'wrong bounded prefix count')
        rid = int(common['request'])
        need(rid > last_request, 'stale request ID exposed')
        last_request = rid
        record = dict(case=case, headers=headers, files=files, lifecycle=common)
        if statehash_record is not None:
            record['statehash'] = statehash_record
        results.append(record)
        checkpoint()

    c = connect()
    try:
        config_line = one(c, 'config')
        cfg = fields(config_line)
        need(cfg.get('rev') == a.source_sha, 'loaded engine revision differs from --source-sha')
        trace_cap = int(cfg.get('trace_tokens', 0))
        dies = int(cfg['dies'])
        maxt = int(cfg['maxt'])
        need(trace_cap >= 4 and dies >= 1 and maxt >= 80,
             'test needs trace window>=4, dies>=1 and capacity>=80')
        pair_enabled = cfg.get('two_position') == '1'
        manifest = dict(
            hostname=socket.gethostname(), source_sha=a.source_sha, config=config_line,
            cubin_sha256=sha(cfg['cubin']), socket=a.socket, device_fault_injected=False,
            pair_check_requested=a.pair_check, two_position=pair_enabled,
            pair_check_executed=False, driver_sha256=sha(__file__))
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        need(one(c, 'stop').startswith('ok:'), 'stop reset failed')
        need(one(c, 'snap 1').startswith('ok:'), 'snapshot request failed')
        tokens, done, _ = run(c, 3, [10, 11])
        need(done.startswith('done ') and len(tokens) == 4, 'normal request did not complete')
        collect(c, 'normal', 'completed', 0, 0, 5, 5)
        first_token = tokens[0]
        need(one(c, 'stop ' + str(first_token)).startswith('ok:'), 'stop setup failed')
        tokens, done, _ = run(c, 3, [10, 11])
        need(done.startswith('done ') and tokens == [first_token], 'known stop token was not reached')
        need(len(done.split()) == 8 and done.split()[6] == '1',
             'stop request did not resume the one-position snapshot')
        collect(c, 'stop', 'completed', 0, 1, 5, 2)
    finally:
        c.close()

    c = connect()
    try:
        need(one(c, 'stop').startswith('ok:'), 'stop reset failed')
        protocol.append('> chat 40 10 11')
        c.send('chat 40 10 11')
        first = c.line()
        protocol.append(first)
        need(first.startswith('tok '), 'no token before cancellation')
    finally:
        c.close(abort=True)
    protocol.append('< client disconnected after first token >')

    c = connect()
    try:
        collect(c, 'disconnect', 'aborted', 1, 1, 42)
        tokens, error, _ = run(c, maxt, [10, 11])
        need(not tokens and error.startswith('error:'), 'oversized request was not rejected')
        collect(c, 'argument-error', 'argument_error', 0, 0, maxt + 2, 0, error=True)
        missing = '/tmp/flash-trace-missing-' + uuid.uuid4().hex + '.bin'
        need(one(c, 'img ' + missing + ' 0').startswith('ok:'), 'image fixture setup failed')
        tokens, error, _ = run(c, 3, [10, 11])
        need(not tokens and error.startswith('error:'), 'missing image was not rejected')
        collect(c, 'image-error', 'image_error', 0, 0, 5, 0, error=True)
        need(one(c, 'stop').startswith('ok:'), 'stop reset failed')
        tokens, done, _ = run(c, 3, [10, 11])
        need(done.startswith('done ') and len(tokens) == 4, 'recovery request failed')
        collect(c, 'recovery', 'completed', 0, 0, 5, 5)

        if a.pair_check and pair_enabled:
            pair_ids = [20, 21, 22, 23]
            need(one(c, 'snap 1').startswith('ok:'), 'paired snapshot request failed')
            pair_tokens, done, _ = run(c, 3, pair_ids)
            need(done.startswith('done ') and len(pair_tokens) == 4, 'paired request did not complete')
            need(len(done.split()) == 8 and done.split()[6:8] == ['0', '1'],
                 'paired request did not create a fresh one-position snapshot')
            state = statehash(c, 'pair-first')
            collect(c, 'pair-first', 'completed', 0, 0, 7, 7,
                    first_position=3, statehash_record=state)
            resumed_tokens, done, _ = run(c, 3, pair_ids)
            need(done.startswith('done ') and resumed_tokens == pair_tokens,
                 'paired snapshot resume changed output tokens')
            need(len(done.split()) == 8 and done.split()[6] == '1',
                 'paired request did not resume the one-position snapshot')
            state = statehash(c, 'pair-resume')
            collect(c, 'pair-resume', 'completed', 0, 0, 7, 7,
                    first_position=3, statehash_record=state)
            manifest['pair_check_executed'] = True
        elif a.pair_check:
            raise AssertionError('--pair-check requires config two_position=1')
    finally:
        c.close()
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    checkpoint('PASS')
    print('PASS: normal, stop/resume, disconnect, argument error, image error, recovery'
          + ('; paired prefill/resume' if manifest['pair_check_executed'] else ''))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
