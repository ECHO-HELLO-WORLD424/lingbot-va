# Composing OOD Evaluation Tasks for RoboTwin

This document covers everything needed to implement new composed tasks in the next phase.
It was written after investigating three broken OOD tasks (`shake_then_place_bottle`,
`dump_bin_then_sort_by_color`, `unpack_then_rank`) and partially implementing three
replacements (`handover_then_hang_mug`, `click_bell_then_sort_blocks`, `stamp_then_stack_bowls`).

---

## 0. TODOs

### Tasks that are abnormal

These tasks are quite abnormal and produces huge amount of error log on evaluation. There are two types of failure pattern:

- **Can have occasional solvable tasks**: This means that at least *some* random seed can produce a task that is solvable by the planner. In this case, log will show huge amount of `planner` errors with one or two normal test results mixed in it.
  > Note: however this often implies tricky edge cases so getting zero on these tasks does not necessarily shows that the robot is incapable.
- **Almost all random seed leads to unsolvable task**: This means that evaluation process will stuck because planner can't calculate valid ground truth. In this case, the model is not actually tested.
  
  > Note: "planner can't calculate valid ground truth" does not mean that the task is unsolvable for human/robot. 
  > It's just that the deterministic solver can't solve it thus we don't have the ground truth for evaluation.

```bash
  "stack_then_scan"                       # Can have occasional solvable tasks
  "shake_then_place_bottle"               # Almost all random seed leads to unsolvable task
  "rotate_qrcode_then_scan"               # Can have occasional solvable tasks
  "open_laptop_then_place_object_inside"  # Can have occasional solvable tasks
  "dump_bin_then_sort_by_color"           # Almost all random seed leads to unsolvable task
  "press_stapler_while_holding"           # Can have occasional solvable tasks
  "unpack_then_rank"                      # Almost all random seed leads to unsolvable task
  "place_dual_shoes_then_hang_mug"        # Can have occasional solvable tasks
  "fill_then_shake_then_move_to_pot"      # Can have occasional solvable tasks
```

### Partially implemented replacement tasks (ready to fix in next phase)

Three replacement tasks were written to substitute the three broken OOD tasks.
They pass the no-timeout and instruction-generation checks but need spatial layout
tuning to achieve consistent `plan_success`:

| New task                      | Replaces                      | Status                | Remaining issue                                           |
| ----------------------------- | ----------------------------- | --------------------- | --------------------------------------------------------- |
| `handover_then_hang_mug`      | `shake_then_place_bottle`     | No timeouts, instr OK | `plan_success=0` — mug hanging fails with mic on table    |
| `click_bell_then_sort_blocks` | `dump_bin_then_sort_by_color` | **PASSES** validator  | —                                                         |
| `stamp_then_stack_bowls`      | `unpack_then_rank`            | No timeouts, instr OK | `plan_success=0` — bowl stacking fails with seal on table |

For `handover_then_hang_mug` and `stamp_then_stack_bowls`, the fix is likely one of:

- Replace the Stage 2 task with something simpler (fewer move steps, single arm)
- Make Stage 1 objects `is_static=True` after Stage 1 completes so they don't count as obstacles
- Use `table_height_bias` to give more vertical clearance (see `place_dual_shoes_then_hang_mug`)

---

## 1. How to Write a New Composed Task

A composed task combines two existing in-distribution tasks into a single `play_once()`.
The pattern is always: **Stage 1 objects in front area → Stage 2 objects in back area →
`back_to_origin()` between stages**.

### File locations

| Artifact         | Path                                                     |
| ---------------- | -------------------------------------------------------- |
| Task class       | `robotwin/envs/<task_name>.py`                           |
| Instruction JSON | `robotwin/description/task_instruction/<task_name>.json` |
| Step limit entry | `robotwin/task_config/_eval_step_limit.yml`              |

The class name must exactly match the filename (no `.py`).

### Skeleton

```python
from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *
import numpy as np

class stage1_then_stage2(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)          # no table_xy_bias

    def load_actors(self):
        # --- Stage 1 objects: front area (ylim < 0) ---
        # Place on clearly one side: xlim = [0.1, 0.25] or [-0.25, -0.1]
        # Never use ylim_prop=True when ylim is all-positive or all-negative
        self.obj_a = create_actor(...)
        self.add_prohibit_area(self.obj_a, padding=0.07)

        # --- Stage 2 objects: back area (ylim > 0.05) ---
        self.obj_b = create_actor(...)
        self.add_prohibit_area(self.obj_b, padding=0.07)

        # Protect the front area from cluttered-table spawning
        self.prohibited_area.append([-0.3, -0.25, 0.3, 0.0])

    def play_once(self):
        # === Stage 1 ===
        arm = ArmTag("right" if self.obj_a.get_pose().p[0] > 0 else "left")
        self.move(self.grasp_actor(self.obj_a, arm_tag=arm, pre_grasp_dis=0.1))
        # ... stage 1 moves ...

        # Reset before stage 2
        self.move(self.back_to_origin(ArmTag("left")))
        self.move(self.back_to_origin(ArmTag("right")))

        # === Stage 2 ===
        # ... stage 2 moves ...

        self.info["info"] = {"{A}": "...", "{B}": "..."}   # must match JSON
        return self.info

    def check_success(self):
        return stage1_ok and stage2_ok
```

### Instruction JSON format

```json
{
  "full_description": "...",
  "schema": "{A} notifies X, {B} notifies Y",
  "preference": "num of words should not exceed 15",
  "seen": ["Do {A} then {B}.", "..."],
  "unseen": ["First do {A}, then {B}.", "..."]
}
```

**Critical**: the placeholder keys in `info["info"]` (e.g. `{A}`, `{B}`) must exactly match
those used in the JSON templates. A mismatch causes `generate_episode_descriptions` to return
an empty list, which crashes the eval with `IndexError: list index out of range`.

### Step limit

Add to `robotwin/task_config/_eval_step_limit.yml`:

```yaml
stage1_then_stage2: 1500   # sum of both base task limits + ~200 buffer
```

---

## 2. Why Combining Arbitrary Tasks Usually Fails

The planner in RoboTwin (`left_plan_path` / `right_plan_path`) uses **interpolative screw
motion** — it plans a straight-line path in joint space and rejects it if any waypoint
collides with the scene. It does **not** do obstacle-avoidance planning.

This means every extra object on the table increases the chance that a straight-line joint
path clips it, setting `plan_success = False`. Composed tasks have twice as many objects,
so failure rates multiply.

### Specific failure modes observed

| Failure                                      | Cause                                                                 | Example task                                                                    |
| -------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `plan_success = False` on every seed         | Too many objects; straight-line paths always collide                  | `handover_then_hang_mug` with mic + mug + rack all present                      |
| Infinite hang during `play_once`             | Planner called in a loop with no timeout; never returns               | `dump_bin_then_sort_by_color`, `unpack_then_rank`                               |
| Infinite hang during `load_actors`           | `while` loop sampling random poses can't satisfy constraints          | `ylim_prop=True` with all-positive ylim; `while abs(x) < 0.15` with narrow xlim |
| `AssertionError: target_pose cannot be None` | `choose_grasp_pose` returns `None` when no contact point is reachable | `shake_then_place_bottle`                                                       |
| `IndexError` in instruction generation       | Info dict keys don't match JSON placeholders                          | `shake_then_place_bottle` (after the planner issue)                             |

### Root cause of `together_move_to_pose` hangs

`_base_task.py:850` has an unbounded `while` loop:

```python
while now_left_id < left_n_step or now_right_id < right_n_step:
    ...
    self.scene.step()
```

If the planner returns a path but the simulation physics diverges (objects flying, joints
locking), the counters never reach `n_step` and the process hangs forever. There is no
timeout or iteration cap.

### What makes a composition safe

- **Strict spatial separation**: Stage 1 objects in front (y < 0), Stage 2 in back (y > 0.05).
  Objects on opposite x-sides further reduce conflicts.
- **Simple base tasks**: Tasks with few move steps and no dual-arm coordination are safer.
  `click_bell` (2 moves) + `place_mouse_pad` (3 moves) works. `hanging_mug` (8 moves) +
  anything tends to fail.
- **No `table_xy_bias`**: Shifting the table changes all z-coordinates and makes the planner
  more likely to find invalid paths.
- **`back_to_origin()` between stages**: Clears arm state so Stage 2 starts from a known pose.
- **Static objects after use**: If Stage 1 leaves a dynamic object on the table, it becomes
  an obstacle for Stage 2. Making it `is_static=True` after Stage 1 would help (not yet
  implemented in the base class).

---

## 3. How to Test if a New Task is Valid

A validation script is at `robotwin/validate_tasks.py`. It runs each task in a child process
with a hard timeout, catching both exceptions and infinite hangs.

### Usage

```bash
# From the RoboTwin directory, using the lingbot-va venv if you installed the dependencies using uv:
# Just run with `python` if you use the Dockerfile
cd /path/to/RoboTwin
python \
    validate_tasks.py \
    --tasks my_new_task another_task \
    --seeds 20 \
    --timeout 180
```

### What it checks per seed

| Column  | Meaning                                                                |
| ------- | ---------------------------------------------------------------------- |
| `setup` | `setup_demo()` + `load_actors()` completed without hanging             |
| `play`  | `play_once()` returned without exception                               |
| `plan`  | `env.plan_success == True` after `play_once()`                         |
| `succ`  | `check_success()` returned `True`                                      |
| `instr` | `generate_episode_descriptions()` returned non-empty seen instructions |
| `⏱`     | Process killed after timeout — infinite loop detected                  |

### Pass criteria (built into the script)

- No more than 50% of seeds timeout
- At least 1 seed achieves `plan_success`
- If any seed completes `play_once()`, at least 1 must generate valid instructions

### Recommended workflow

1. Run the two **base tasks** alone first to confirm they pass individually.
2. Run the **composed task** with `--seeds 20 --timeout 180`.
3. If timeouts appear: fix `load_actors` (check for infinite `while` loops, bad `ylim_prop`).
4. If `plan_success = 0/N` with no timeouts: the planner is consistently failing — objects
   are too close together. Widen the spatial separation or choose simpler base tasks.
5. If `instr = False`: check that `info["info"]` keys match the JSON placeholders exactly.

### Common `load_actors` bugs to check before running

```python
# BAD: ylim_prop=True forces y<0 when abs(x)<0.15, but ylim is all-positive → infinite loop
rand_pose(xlim=[-0.3, 0.3], ylim=[0.05, 0.2], ylim_prop=True)

# BAD: xlim range too narrow for the abs(x) constraint → near-infinite loop
rand_pose(xlim=[-0.2, 0.2], ...)
while abs(rand_pos.p[0]) < 0.15:   # only 0.05 of range satisfies this
    ...

# GOOD: sample directly from the valid range
side = np.random.choice([-1, 1])
rand_pose(xlim=[side * 0.15, side * 0.25], ...)

# BAD: xlim with negative side gives reversed limits
rand_pose(xlim=[side * 0.25, side * 0.1], ...)  # [-0.25, -0.1] ok, but [0.25, 0.1] invalid

# GOOD: always sort
xlim = sorted([side * 0.1, side * 0.25])
rand_pose(xlim=xlim, ...)
```

---

## 4. Entry Points, Integration Points, and Test Code

### Where to add a new task

```
RoboTwin/
├── envs/
│   ├── my_new_task.py              ← 1. Task class (new file)
│   └── __init__.py                 ← 2. Register: add import + class to TASK_LIST
├── description/
│   └── task_instruction/
│       └── my_new_task.json        ← 3. Instruction templates (new file)
├── task_config/
│   └── _eval_step_limit.yml        ← 4. Add step limit entry
└── validate_tasks.py               ← 5. Test script (already exists)
```

Check `envs/__init__.py` for the exact registration pattern — some versions use a dict,
others use dynamic import. The eval script imports tasks by name via `importlib`.

### Where the eval picks up new tasks

The OOD eval launch script is:

```
lingbot-va/evaluation/robotwin/launch_ood_eval.sh
```

Add the new task name to the task list in that script. The eval client is:

```
lingbot-va/evaluation/robotwin/eval_polict_client_openpi.py
```

Key function: `eval_policy()` — it calls `setup_demo` + `play_once` in a loop, skipping
seeds where `plan_success=False` or an exception is raised, until `test_num` successes
are collected.

### Instruction generation entry point

```
RoboTwin/description/utils/generate_episode_instructions.py
  └── generate_episode_descriptions(task_name, episodes, max_descriptions)
        └── load_task_instructions(task_name)   # reads the JSON
        └── filter_instructions(instructions, episode)
        └── replace_placeholders(instruction, episode)
```

The `episode` dict passed in is `play_once()["info"]` — the keys must match JSON placeholders.

### Test script

```
RoboTwin/validate_tasks.py
```

Run from the `RoboTwin/` directory with the `lingbot-va` venv (which has SAPIEN):

```bash
cd /path/to/RoboTwin
python \
    validate_tasks.py --tasks <task_name> --seeds 20 --timeout 180
```
