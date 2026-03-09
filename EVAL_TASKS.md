# LingBot-VA Evaluation Tasks

LingBot-VA is evaluated on the **RoboTwin 2.0** benchmark, a bimanual robotic manipulation platform using a dual-arm robot (aloha-agilex embodiment). The benchmark includes **50 tasks** with two evaluation modes:

- **Easy (demo_clean)**: no domain randomization, clean backgrounds
- **Hard (demo_randomized)**: random backgrounds, cluttered tables, random lighting

Each task runs for 100 episodes per evaluation. Performance is measured by **Success Rate (SR)**.

LingBot-VA achieves **92.9% Easy SR** and **91.6% Hard SR** averaged over all 50 tasks, surpassing previous state-of-the-art.

---

## Full Task List (50 Tasks)

The tasks are listed below with their maximum allowed steps (step limit). Tasks are categorized by manipulation type.

### Object Placement

| Task | Step Limit | Description |
|------|-----------|-------------|
| `place_a2b_left` | 400 | Place an object from position A to position B using the left arm |
| `place_a2b_right` | 400 | Place an object from position A to position B using the right arm |
| `place_bread_basket` | 700 | Place bread into a basket |
| `place_bread_skillet` | 500 | Place bread into a skillet |
| `place_burger_fries` | 500 | Place a burger and fries |
| `place_can_basket` | 700 | Place a can into a basket |
| `place_cans_plasticbox` | 800 | Place cans into a plastic box |
| `place_container_plate` | 400 | Place a container onto a plate |
| `place_dual_shoes` | 600 | Place a pair of shoes |
| `place_empty_cup` | 500 | Place an empty cup |
| `place_fan` | 400 | Place a fan |
| `place_mouse_pad` | 400 | Place a mouse onto a pad |
| `place_object_basket` | 700 | Place an object into a basket |
| `place_object_scale` | 400 | Place an object onto a scale |
| `place_object_stand` | 400 | Place an object onto a stand |
| `place_phone_stand` | 400 | Place a phone onto a stand |
| `place_shoe` | 500 | Place a single shoe |

### Stacking

| Task | Step Limit | Description |
|------|-----------|-------------|
| `stack_blocks_two` | 800 | Stack two blocks on top of each other |
| `stack_blocks_three` | 1200 | Stack three blocks on top of each other |
| `stack_bowls_two` | 900 | Stack two bowls |
| `stack_bowls_three` | 1200 | Stack three bowls |

### Sorting / Ranking

| Task | Step Limit | Description |
|------|-----------|-------------|
| `blocks_ranking_rgb` | 1200 | Rank blocks by color (RGB ordering) |
| `blocks_ranking_size` | 1200 | Rank blocks by size |

### Picking

| Task | Step Limit | Description |
|------|-----------|-------------|
| `pick_diverse_bottles` | 400 | Pick a bottle from a diverse set |
| `pick_dual_bottles` | 400 | Pick two bottles simultaneously |
| `grab_roller` | 400 | Grab a roller object |

### Moving / Repositioning

| Task | Step Limit | Description |
|------|-----------|-------------|
| `adjust_bottle` | 400 | Adjust the orientation or position of a bottle |
| `move_can_pot` | 400 | Move a can into a pot |
| `move_pillbottle_pad` | 400 | Move a pill bottle onto a pad |
| `move_playingcard_away` | 400 | Move a playing card away from a location |
| `move_stapler_pad` | 400 | Move a stapler onto a pad |

### Handover / Bimanual Transfer

| Task | Step Limit | Description |
|------|-----------|-------------|
| `handover_block` | 800 | Hand over a block from one arm to the other |
| `handover_mic` | 600 | Hand over a microphone from one arm to the other |

### Articulated / Mechanism Interaction

| Task | Step Limit | Description |
|------|-----------|-------------|
| `open_laptop` | 700 | Open a laptop lid |
| `open_microwave` | 1500 | Open a microwave door |
| `turn_switch` | 400 | Toggle a switch |
| `click_alarmclock` | 400 | Press the button on an alarm clock |
| `click_bell` | 400 | Click a bell |
| `press_stapler` | 400 | Press down on a stapler |
| `stamp_seal` | 400 | Press a seal stamp onto a surface |

### Container / Bin Operations

| Task | Step Limit | Description |
|------|-----------|-------------|
| `dump_bin_bigbin` | 600 | Dump contents of a bin into a larger bin |
| `put_bottles_dustbin` | 1700 | Put multiple bottles into a dustbin |
| `put_object_cabinet` | 700 | Put an object into a cabinet |

### Hanging / Suspension

| Task | Step Limit | Description |
|------|-----------|-------------|
| `hanging_mug` | 900 | Hang a mug on a hook |
| `lift_pot` | 400 | Lift a pot |

### Manipulation with Constraints

| Task | Step Limit | Description |
|------|-----------|-------------|
| `rotate_qrcode` | 400 | Rotate a QR code to a target orientation |
| `scan_object` | 500 | Scan an object (move it in front of a scanner) |
| `shake_bottle` | 700 | Shake a bottle vertically |
| `shake_bottle_horizontally` | 700 | Shake a bottle horizontally |

---

## Real-World Evaluation Tasks

Six real-world tasks across three categories:

### Long-Horizon Tasks
| Task | Description |
|------|-------------|
| Make Breakfast | Multi-step breakfast preparation sequence |
| Pick Screws | Pick and place small screws |

### Precision Tasks
| Task | Description |
|------|-------------|
| Insert Tube | Insert a tube into a tight fitting |
| Unpack Delivery | Unpack items from a delivery box |

### Deformable & Articulated Object Manipulation
| Task | Description |
|------|-------------|
| Fold Clothes | Fold a piece of clothing |
| Fold Pants | Fold a pair of pants |

---

## Simulation Benchmark: LIBERO

LingBot-VA is also evaluated on the **LIBERO** benchmark across four suites:

| Suite | Focus |
|-------|-------|
| LIBERO-Spatial | Spatial relationship understanding |
| LIBERO-Object | Object-centric manipulation |
| LIBERO-Goal | Goal-conditioned tasks |
| LIBERO-Long | Long-horizon sequential tasks |

LingBot-VA achieves **98.5% average SR** across all four LIBERO suites.
