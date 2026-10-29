"""The memory room of a container is the limit minus the working set (E3c).

`memory.current` (v2) and `memory.usage_in_bytes` (v1) count the page cache, and the kernel takes the inactive file pages
back when a process needs memory. The used memory is therefore `current - inactive_file`, the rule of cAdvisor and the
kubelet (Docker subtracts the same value). The tests build a fake cgroup root, so they do not depend on the host.
"""
import pytest

from topopt import cli, environment
from topopt.verify import memory

GIB = 2 ** 30

# Trimmed copy of /sys/fs/cgroup/memory.stat of a container with a warm page cache (cgroup v2, Linux 7.2, 2026-10-21; 64 GiB
# limit). memory.current of that moment was about anon + file + kernel, the value that the CURRENT constant holds.
V2_STAT = """anon 4483796992
file 57850232832
kernel 3828105216
shmem 327680
file_mapped 581185536
inactive_anon 541204480
active_anon 3951333376
inactive_file 56895205376
active_file 954744832
unevictable 0
slab_reclaimable 3779959400
"""
LIMIT = 64 * GIB
CURRENT = 4483796992 + 57850232832 + 3828105216
INACTIVE = 56895205376

# cgroup v1 memory.stat (kernel docs, cgroup-v1/memory): the lines of this cgroup first, then the `total_` lines with the
# children. The file is typed from the documented format: no cgroup v1 host was available here.
V1_STAT = """cache 900
rss 100
inactive_anon 10
active_anon 90
inactive_file 400
active_file 300
total_cache 1500
total_rss 200
total_inactive_anon 20
total_active_anon 180
total_inactive_file 700
total_active_file 600
"""


def write(root, name, text):
    f = root / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text)


def v2(root, current=CURRENT, stat=V2_STAT, limit=LIMIT):
    write(root, "memory.max", f"{limit}\n")
    write(root, "memory.current", f"{current}\n")
    if stat is not None:
        write(root, "memory.stat", stat)


def v1(root, usage=2000, stat=V1_STAT, limit=4000):
    write(root, "memory/memory.limit_in_bytes", f"{limit}\n")
    write(root, "memory/memory.usage_in_bytes", f"{usage}\n")
    if stat is not None:
        write(root, "memory/memory.stat", stat)


def fake_host(monkeypatch, root, host_avail, host_total=128 * GIB):
    monkeypatch.setattr(cli, "CGROUP_ROOT", root)
    monkeypatch.setattr(cli, "host_ram_bytes", lambda: host_total)
    monkeypatch.setattr(environment, "host_ram_bytes", lambda: host_total)
    monkeypatch.setattr(environment, "_available_ram_bytes", lambda: host_avail)


# --------------------------------------------------------------------------- #
# cgroup_limits: the working set
# --------------------------------------------------------------------------- #

def test_v2_used_memory_is_current_minus_inactive_file(tmp_path):
    v2(tmp_path)
    lim = cli.cgroup_limits(tmp_path)
    assert lim["source"] == "v2" and lim["memory_bytes"] == LIMIT
    assert lim["memory_used_bytes"] == CURRENT - INACTIVE
    assert lim["memory_current_bytes"] == CURRENT and lim["inactive_file_bytes"] == INACTIVE
    assert lim["facts"] == []


def test_v2_the_raw_room_was_small_and_the_working_set_room_is_large(tmp_path):
    """The incident: a warm cache left 2.4 GiB of room by memory.current, and 55 GiB by the working set."""
    v2(tmp_path)
    lim = cli.cgroup_limits(tmp_path)
    assert (LIMIT - lim["memory_current_bytes"]) / GIB < 3
    assert (lim["memory_bytes"] - lim["memory_used_bytes"]) / GIB > 55


def test_v2_used_memory_is_never_below_zero_and_a_fact_names_both_values(tmp_path):
    v2(tmp_path, current=1000, stat="inactive_file 5000\n")     # memory.current and memory.stat are two reads
    lim = cli.cgroup_limits(tmp_path)
    assert lim["memory_used_bytes"] == 0 and lim["memory_current_bytes"] == 1000 and lim["inactive_file_bytes"] == 5000
    assert len(lim["facts"]) == 1
    fact = lim["facts"][0]
    assert "memory.current" in fact and "inactive_file" in fact and "memory.stat" in fact
    assert "1000 bytes" in fact and "5000 bytes" in fact and "one after the other" in fact and "set to 0" in fact


def test_v2_inactive_file_equal_to_current_gives_zero_and_no_fact(tmp_path):
    v2(tmp_path, current=1000, stat="inactive_file 1000\n")
    lim = cli.cgroup_limits(tmp_path)
    assert lim["memory_used_bytes"] == 0 and lim["facts"] == []


def test_v2_an_inactive_file_of_zero_is_a_value_not_a_missing_line(tmp_path):
    v2(tmp_path, current=1000, stat="anon 1000\ninactive_file 0\n")
    lim = cli.cgroup_limits(tmp_path)
    assert lim["memory_used_bytes"] == 1000 and lim["inactive_file_bytes"] == 0 and lim["facts"] == []


@pytest.mark.parametrize("stat", [None, "", "anon 1\nfile 2\n", "active_file 7\ntotal_inactive_file 9\n", "inactive_file x\n",
                                  "inactive_file\n", "inactive_file -5\n"])
def test_v2_without_a_usable_inactive_file_line_the_used_memory_is_current_and_a_fact_says_so(tmp_path, stat):
    v2(tmp_path, current=3000, stat=stat)
    lim = cli.cgroup_limits(tmp_path)
    assert lim["memory_used_bytes"] == 3000 and lim["memory_current_bytes"] == 3000 and lim["inactive_file_bytes"] is None
    assert len(lim["facts"]) == 1
    assert "memory.stat" in lim["facts"][0] and "inactive_file" in lim["facts"][0] and "memory.current" in lim["facts"][0]
    assert "memory_used_bytes" not in lim["facts"][0]            # a key name of the code is not a name that the reader sees


def test_the_fallback_fact_speaks_of_a_room_only_when_there_is_a_limit(tmp_path):
    v2(tmp_path, current=3000, stat=None)
    assert "the room can look smaller" in cli.cgroup_limits(tmp_path)["facts"][0]
    write(tmp_path, "memory.max", "max\n")                       # no limit: there is no room to look small
    lim = cli.cgroup_limits(tmp_path)
    assert lim["memory_bytes"] is None and len(lim["facts"]) == 1
    assert "room" not in lim["facts"][0] and "page cache" in lim["facts"][0]
    v1(tmp_path / "v1", usage=2000, stat=None, limit=2 ** 61)     # v1 "no limit" is a huge number
    lim = cli.cgroup_limits(tmp_path / "v1")
    assert lim["memory_bytes"] is None and "room" not in lim["facts"][0]


def test_v2_the_key_matches_the_whole_name(tmp_path):
    """`active_file` and `total_inactive_file` are other lines: only `inactive_file` counts in v2."""
    v2(tmp_path, current=3000, stat="active_file 100\ntotal_inactive_file 200\ninactive_file 300\n")
    assert cli.cgroup_limits(tmp_path)["memory_used_bytes"] == 2700


def test_v1_used_memory_is_usage_minus_total_inactive_file(tmp_path):
    v1(tmp_path, usage=2000)
    lim = cli.cgroup_limits(tmp_path)
    assert lim["source"] == "v1" and lim["memory_bytes"] == 4000
    assert lim["inactive_file_bytes"] == 700                       # total_inactive_file (the children too), not inactive_file 400
    assert lim["memory_used_bytes"] == 1300 and lim["memory_current_bytes"] == 2000 and lim["facts"] == []


def test_v1_used_memory_is_never_below_zero_and_a_fact_names_both_values(tmp_path):
    v1(tmp_path, usage=500)                                       # total_inactive_file is 700
    lim = cli.cgroup_limits(tmp_path)
    assert lim["memory_used_bytes"] == 0 and lim["memory_current_bytes"] == 500 and lim["inactive_file_bytes"] == 700
    assert len(lim["facts"]) == 1
    fact = lim["facts"][0]
    assert "memory.usage_in_bytes" in fact and "total_inactive_file" in fact and "500 bytes" in fact and "700 bytes" in fact


@pytest.mark.parametrize("stat", [None, "inactive_file 400\ncache 1\n"])
def test_v1_without_total_inactive_file_the_used_memory_is_usage_and_a_fact_says_so(tmp_path, stat):
    v1(tmp_path, usage=2000, stat=stat)
    lim = cli.cgroup_limits(tmp_path)
    assert lim["memory_used_bytes"] == 2000 and lim["inactive_file_bytes"] is None
    assert len(lim["facts"]) == 1 and "total_inactive_file" in lim["facts"][0] and "memory.usage_in_bytes" in lim["facts"][0]


def test_no_current_value_gives_no_used_memory_and_no_fact(tmp_path):
    write(tmp_path, "memory.max", "max\n")
    lim = cli.cgroup_limits(tmp_path)
    assert lim["memory_used_bytes"] is None and lim["memory_current_bytes"] is None and lim["facts"] == []


def test_no_cgroup_files_give_empty_facts(tmp_path):
    lim = cli.cgroup_limits(tmp_path)
    assert lim["source"] is None and lim["memory_used_bytes"] is None and lim["facts"] == []


def test_the_real_cgroup_of_this_host_follows_the_rule():
    """On the real /sys/fs/cgroup: used = max(0, current - inactive_file), or current with a fact. Skipped without a cgroup."""
    lim = cli.cgroup_limits()
    if lim["memory_current_bytes"] is None:
        pytest.skip("no readable cgroup memory.current on this host")
    cur, inactive = lim["memory_current_bytes"], lim["inactive_file_bytes"]
    if inactive is None:
        assert lim["memory_used_bytes"] == cur and lim["facts"]
    else:
        assert lim["memory_used_bytes"] == max(0, cur - inactive) and lim["facts"] == []
    assert 0 <= lim["memory_used_bytes"] <= cur


# --------------------------------------------------------------------------- #
# environment._ram and the verify limit: the room
# --------------------------------------------------------------------------- #

def test_ram_room_is_the_limit_minus_the_working_set(tmp_path, monkeypatch):
    v2(tmp_path)
    fake_host(monkeypatch, tmp_path, host_avail=80 * GIB)
    ram = environment._ram()
    assert ram["cgroup_room_bytes"] == LIMIT - (CURRENT - INACTIVE)
    assert ram["available_bytes"] == ram["cgroup_room_bytes"] and ram["available_bytes"] / GIB > 55
    assert ram["cgroup_used_bytes"] == CURRENT - INACTIVE and ram["cgroup_current_bytes"] == CURRENT
    assert ram["cgroup_inactive_file_bytes"] == INACTIVE
    assert ram["cgroup_limit_bytes"] == LIMIT and ram["total_bytes"] == LIMIT and ram["cgroup_facts"] == []


def test_ram_the_host_value_wins_when_it_is_lower_than_the_room(tmp_path, monkeypatch):
    v2(tmp_path)
    fake_host(monkeypatch, tmp_path, host_avail=10 * GIB)
    ram = environment._ram()
    assert ram["available_bytes"] == 10 * GIB and ram["cgroup_room_bytes"] > 10 * GIB


def test_ram_without_a_host_value_uses_the_room(tmp_path, monkeypatch):
    v1(tmp_path, usage=2000)
    fake_host(monkeypatch, tmp_path, host_avail=None, host_total=None)
    ram = environment._ram()
    assert ram["available_bytes"] == 4000 - 1300 and ram["total_bytes"] == 4000 and ram["cgroup_source"] == "v1"


def test_ram_room_without_memory_stat_is_limit_minus_current_and_the_fact_is_shown(tmp_path, monkeypatch):
    v2(tmp_path, stat=None)
    fake_host(monkeypatch, tmp_path, host_avail=80 * GIB)
    ram = environment._ram()
    assert ram["available_bytes"] == LIMIT - CURRENT and ram["cgroup_used_bytes"] == CURRENT
    assert len(ram["cgroup_facts"]) == 1 and "memory.stat" in ram["cgroup_facts"][0]


def test_ram_the_room_is_never_below_zero(tmp_path, monkeypatch):
    v2(tmp_path, current=LIMIT + 5, stat="anon 1\n")
    fake_host(monkeypatch, tmp_path, host_avail=80 * GIB)
    ram = environment._ram()
    assert ram["cgroup_room_bytes"] == 0 and ram["available_bytes"] == 0


def test_ram_inactive_file_above_current_shows_both_values_and_the_fact(tmp_path, monkeypatch):
    """The race case: the used memory is 0 and the room is the whole limit, so `environment` must show why."""
    v2(tmp_path, current=6 * GIB, stat=f"inactive_file {7 * GIB}\n", limit=8 * GIB)
    fake_host(monkeypatch, tmp_path, host_avail=80 * GIB)
    ram = environment._ram()
    assert ram["cgroup_used_bytes"] == 0 and ram["cgroup_room_bytes"] == 8 * GIB and ram["available_bytes"] == 8 * GIB
    assert ram["cgroup_current_bytes"] == 6 * GIB and ram["cgroup_inactive_file_bytes"] == 7 * GIB
    assert len(ram["cgroup_facts"]) == 1 and str(7 * GIB) in ram["cgroup_facts"][0]


def test_no_cgroup_gives_the_host_values_and_no_room(tmp_path, monkeypatch):
    fake_host(monkeypatch, tmp_path, host_avail=20 * GIB)
    ram = environment._ram()
    assert ram["available_bytes"] == 20 * GIB and ram["cgroup_room_bytes"] is None and ram["cgroup_facts"] == []


def test_verify_limit_sees_the_room_of_the_working_set(tmp_path, monkeypatch):
    """The path of `topopt verify`: no `--max-gb` and a warm page cache. The limit is the 55 GiB room, not 2.4 GiB."""
    v2(tmp_path)
    fake_host(monkeypatch, tmp_path, host_avail=80 * GIB)
    lim = memory.limits(None)
    assert lim["available_bytes"] == lim["limit_bytes"] == LIMIT - (CURRENT - INACTIVE) and lim["limit_source"] == "available RAM"
    assert memory.limits(8.0)["limit_source"] == "--max-gb"             # 8 GB is below the room
