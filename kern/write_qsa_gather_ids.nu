#!/usr/bin/env nu

def captured-rows [nv: int count: int iterations: int seed: int] {
    mut state = $seed
    mut rows = []
    for $iteration in 0..<$iterations {
        mut swaps = {}
        mut row = []
        for $index in 0..<$count {
            $state = (($state * 1664525 + 1013904223) mod 4294967296)
            let index_key = ($index | into string)
            let pick = $index + ($state mod ($nv - $index))
            let pick_key = ($pick | into string)
            let at_index = ($swaps | get -o $index_key | default $index)
            let at_pick = ($swaps | get -o $pick_key | default $pick)
            $swaps = ($swaps | upsert $index_key $at_pick | upsert $pick_key $at_index)
            $row = ($row ++ [$at_pick])
        }
        $rows = ($rows ++ [($row | sort)])
    }
    $rows
}

def main [nv: int count: int iterations: int seed: int path: string] {
    let rows = (captured-rows $nv $count $iterations $seed)
    let bytes = ($rows | each {|row| $row | each {|value| bytes build ($value mod 256) (($value // 256) mod 256) (($value // 65536) mod 256) (($value // 16777216) mod 256) } | bytes collect } | bytes collect)
    $bytes | save --raw $path
}
