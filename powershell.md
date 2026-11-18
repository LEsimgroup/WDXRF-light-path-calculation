## 两个独立命令

### 命令 1：加载/查看强度分布文件（不运行模拟）

```bash
python -m xrf_sim.view_profile --file ./xrf_sim/spot_profile/wafer_spot_profile-open.npy --save_dir ./sim_output
```

**功能：**

- 显示分布统计信息（shape、峰值、均值、总强度、范围）
- 生成 4 张图：强度热力图、沿 x 剖面、沿 y 剖面、强度直方图
- 保存为 `profile_view.png`

**参数：**
| 参数 | 说明 |
|------|------|
| `--file` | 强度分布文件路径（必填） |
| `--x` | x 范围 `xmin xmax`（mm） |
| `--y` | y 范围 `ymin ymax`（mm） |
| `--save_dir` | 图片保存目录 |

### 命令 2：运行光路模拟（可加载强度分布）

```bash
python -m xrf_sim.debug_tool --crystal flat --profile ./sim_output/test_profile.npy \
    --profile_x_range -25 25 --profile_y_range -25 25 --save_dir ./sim_output
```

**功能：** 加载强度分布 → 按分布采样光斑 → 运行完整光路模拟 → 保存图片 + PC 光子数据

---

### 我做了什么改动

| 文件                | 改动                 |
| ----------------- | ------------------ |
| `view_profile.py` | **新增**：独立的强度分布查看工具 |
| `powershell.md`   | 整理两个命令的用法          |

### 验证结果

- ✅ 命令 1（查看）：成功显示分布信息并保存热力图
- ✅ 命令 2（模拟）：成功加载强度分布并运行光路模拟

两个命令职责分离：**命令 1 用于检查你的强度分布文件是否正确**，**命令 2 用于实际运行光路模拟**。

Made changes.

## 平晶（flat）

```bash
# 基本运行（默认参数）
python -m xrf_sim.debug_tool --crystal flat

# 指定能量、光子数、保存图片
python -m xrf_sim.debug_tool --crystal flat --energy 183 --N 500000 --save_dir ./sim_output

# 启用 Soller 狭缝（平晶通道）
python -m xrf_sim.debug_tool --crystal flat --flatted --save_dir ./sim_output

#B channel new 40窗口
python -m xrf_sim.debug_tool --crystal flat --N 2000000 --flatted --save_dir ./sim_output `
--profile ./xrf_sim/spot_profile/wafer_spot_profile-open.npy `
--energy 183 --d_spacing 8.95e-9 `
--det_dist 90 `
--pc_size 40 20 `
--det_size 21 19 
```


#La channel new 40窗口
python -m xrf_sim.debug_tool --crystal flat --N 10000000 --flatted --save_dir ./sim_output `
--profile ./xrf_sim/spot_profile/wafer_spot_profile-open.npy `
--energy 4647 --d_spacing 0.2013e-9 `
--det_dist 90 `
--pc_size 40 22 `
--det_size 18.6 18.2 `
--fwhm 0.1 `
--cryst_size 30 75


## 柱面 Johann 型弯晶（cylindrical）

> ⚠️ **必须提供 `--ROC`**（曲率半径），否则报错。

```bash
# 基本运行（ROC=100 mm）
python -m xrf_sim.debug_tool --crystal cylindrical --ROC 100

# 指定 ROC 和保存图片
python -m xrf_sim.debug_tool --crystal cylindrical --ROC 150 --N 500000 --save_dir ./sim_output

# P channel仿真
python -m xrf_sim.debug_tool --crystal cylindrical --N 2000000 --profile ./xrf_sim/spot_profile/wafer_spot_profile-open.npy `
    --profile_x_range -30 30 --cryst_size 42 30 `
    --det_size 25 5 --pc_size 40 20 `
    --ROC 150 --fwhm 0.1 --energy 2010 `
    --d_spacing 0.4371e-9 --det_dist 80 `
    --profile_y_range -25 25 --cone 10`
    --save_dir ./sim_output

# La channel柱面仿真


python -m xrf_sim.debug_tool --crystal cylindrical --N 10000000 --profile ./xrf_sim/spot_profile/wafer_spot_profile-open.npy `
    --profile_x_range -30 30 --cryst_size 42 30 `
    --det_size 30 1 --pc_size 32 22 `
    --ROC 150 --fwhm 0.1 --energy 4647 --d_spacing 0.2013e-9 `
    --det_dist 80 `
    --fwhm 0.02`
    --cryst_size 30 70`
    --region_circle 0 0 15`
    --save_dir ./sim_output



#增加锥角限制，提升模拟效率
 python -m xrf_sim.debug_tool --crystal cylindrical --N 1000000 --profile ./sim_output/test_profile.npy --profile_x_range -25 25 --cryst_size 25 42 --det_size 25 5 --pc_size 50 10 --ROC 150 --fwhm 0.1 --energy 2010 --d_spacing 0.4371e-9 --det_dist 80 --profile_y_range -25 25 --cone 10 --save_dir ./sim_output
```

## 对数螺线弯晶（spiral）

> ⚠️ **必须提供 `--a_crystal`**（slit 到晶体中心距离），否则报错。

```bash
# 基本运行（a_crystal=30 mm）
python -m xrf_sim.debug_tool --crystal spiral --a_crystal 30

# 指定 a_crystal 和保存图片
python -m xrf_sim.debug_tool --crystal spiral --a_crystal 30 --save_dir ./sim_output
```

# La channel对数螺线
python -m xrf_sim.debug_tool --crystal spiral --a_crystal 160 --save_dir ./sim_output `
--profile ./xrf_sim/spot_profile/wafer_spot_profile-open.npy `
--N 10000000 `
--pc_size 32 22 `
--det_size 30 1 `
--det_dist 120 `
--energy 4647 --d_spacing 0.2013e-9 `
--fwhm 0.02 `
--cryst_size 30 70 `
--region_circle 0 0 15 

### spiral 的 YZ 投影只显示被反射的光线

spiral 晶体的全部入射光线在 YZ 投影上形成密集红锥，会淹没反射段。
因此**默认只绘制「成功被反射」的那部分光线的完整路径**：

- 入射段：wafer 发光点 → slit → 晶体命中点（红色细线，`Incident (reflected)`）
- 反射段：晶体命中点 → PC 窗口（品红色，`Reflected`）

未被反射 / 未命中 PC 的光线**不绘制**。图像标题会标注 `(reflected paths only)`。

需要看全部入射光线时加 `--yz_full`；其他晶体类型（`flat`/`cylindrical`）
默认仍显示完整光路，可用 `--yz_reflected` 切换。

```bash
# 强制显示完整光路（含全部入射光线）
python -m xrf_sim.debug_tool --crystal spiral --a_crystal 160 --yz_full --save_dir ./sim_output

# flat 只看被反射光线的完整路径
python -m xrf_sim.debug_tool --crystal flat --yz_reflected --save_dir ./sim_output
```


## 常用参数速查

| 参数              | 说明                                 | 默认值               |
| --------------- | ---------------------------------- | ----------------- |
| `--crystal`     | 晶体类型：`flat`/`cylindrical`/`spiral` | `flat`            |
| `--energy`      | 光子能量 (eV)                          | `183`             |
| `--d_spacing`   | 晶面间距 (m)                           | `8.95e-9`         |
| `--det_dist`    | Wafer 到 slit 距离 (mm)               | `124`             |
| `--det_deg`     | 探测器掠射角 (deg)                       | `30`              |
| `--det_size`    | 探测器尺寸 宽 高 (mm)                     | `25 5`            |
| `--cryst_dist`  | 晶体到探测器距离 (mm)                      | `30`              |
| `--cryst_size`  | 晶体尺寸 宽 高 (mm)                      | `30 40`           |
| `--N`           | 光子总数                               | `500000`          |
| `--fwhm`        | 反射率 FWHM (deg)                     | `0.02`            |
| `--spot_radius` | 光斑半径 (mm)                          | `20`              |
| `--ROC`         | 弯晶曲率半径 (mm)                        | 无（cylindrical 必填） |
| `--a_crystal`   | slit 到晶体距离 (mm)                    | 无（spiral 必填）      |
| `--flatted`     | 启用 Soller 狭缝                       | 关                 |
| `--pc_size`     | PC 窗口尺寸 宽 高 (mm)                   | `25 5`            |
| `--region_rect` | 矩形统计区域 `xmin xmax ymin ymax` (mm)，可重复多次 | 无             |
| `--region_circle` | 圆形统计区域 `cx cy r` (mm)，可重复多次      | 无             |
| `--yz_full` / `--yz_reflected` | YZ 投影图光线模式：完整光路 / 只显示被反射光线的完整路径（默认：spiral 后者，其他前者） | 自动 |
| `--save_dir`    | 图片保存目录                             | 无（弹窗）             |

---

## 一个综合示例（弯晶 + 保存）

```bash
python -m xrf_sim.debug_tool --crystal cylindrical --ROC 150 --fwhm 0.1 --energy 2010 --d_spacing 0.4371e-9 --det_dist 80 --det_size 30 5 --profile ./xrf_sim/spot_profile/wafer_spot_profile.npy --save_dir ./sim_output
```

---

## 统计指定区域的 PC 接收光强占比

用 `--region_rect` / `--region_circle` 可以在 wafer 面上指定任意多个区域
（矩形或圆形），输出**每个区域内被 PC 接收到的光强占「全部 PC 接收光强」的比例**。

```bash
# 三个同心圆（内/中/外圈）+ 上下半平面
python -m xrf_sim.debug_tool --crystal flat --N 100000 `
--profile ./xrf_sim/spot_profile/wafer_spot_profile-open.npy `
--profile_x_range -25 25 --profile_y_range -25 25 `
--region_circle 0 0 8 --region_circle 0 0 16 --region_circle 0 0 25 `
--region_rect -25 25 0 25 --region_rect -25 25 -25 0 `
--save_dir ./sim_output
```

**参数格式**

| 参数 | 顺序 | 示例 |
|------|------|------|
| `--region_rect` | `xmin xmax ymin ymax` | `--region_rect -25 25 0 25` |
| `--region_circle` | `cx cy r` | `--region_circle 0 0 8` |

两个参数都可以**重复出现**，每次都新增一个区域。

**输出指标**（`params.md` 表格 + 控制台）

| 指标 | 含义 |
|------|------|
| `占全部PC光强` | **核心指标**：区域内的 PC 命中数 / 全部 PC 命中数 = N_pc(区域) / N_pc_total。回答「该区域贡献了总 PC 信号的百分之多少」 |
| `占全部发射光强` | 区域内的 PC 命中数 / 全部发射光子数 = N_pc(区域) / N_emit_total（等价于该区域对总光强的贡献，互斥铺满时之和 ≈ 总 PC 效率） |
| `占profile强度` | 区域内的 profile 积分强度 / 全部 profile 总强度（纯几何/分布占比，不含晶体/PC 响应） |
| `区域收集效率` | 区域内的 PC 命中数 / 区域内发射光子数（该区域本身的收集效率） |

**自洽性检查**

- 若各区域**互斥且铺满**整个发光区（如上半平面 + 下半平面），
  各区域的「占全部PC光强」之和 = **100%**。
- 若某区域覆盖了全部发光区（如足够大的圆），其值也应为 **100%**。
- 若区域相互重叠（如多个同心圆）或未铺满，之和会不等于 100%，
  程序会在控制台与 `params.md` 中提示。

**输出图**：`debug_<type>_wafer_origins.png` 变为双面板 —— 左侧为 wafer 面
profile 热图 + PC 命中光子的发光原点散点 + 各区域轮廓虚线；
右侧为各区域「占全部 PC 光强比例」的横向柱状图。

---

## 参数扫描

单个参数用一次 `debug_tool` 只能看一个点。要扫一段区间（如 FWHM、探测器角度、
ROC、`a_crystal`、能量……），用**通用扫描脚本** `scan_params.py`。
它基于 `simulate()` 完整管线（**含 PC 窗口**），所以能同时得到
`N_hit_crystal` / `N_reflected` / `N_pc_hit` / `pc_efficiency`。

### 基本用法

```bash
# 一维扫描：FWHM 从 0.01 到 0.2，共 10 点（start:stop:num，含端点）
python -m xrf_sim.experiments.scripts.scan_params --crystal spiral --a_crystal 160 `
--N 200000 `
--sweep "fwhm_deg=0.01:0.2:10"

# 一维扫描：显式取值列表（v1,v2,v3）
python -m xrf_sim.experiments.scripts.scan_params --crystal flat --N 200000 `
--sweep "detector_theta_deg=25,30,35,40"

# 二维网格：--sweep 重复出现，按笛卡尔积展开
python -m xrf_sim.experiments.scripts.scan_params --crystal cylindrical --ROC 150 --N 100000 `
--sweep "detector_theta_deg=28:34:4" `
--sweep "fwhm_deg=0.1,0.5,1.0"
```

`--sweep` 的键名必须是 `SimulationConfig` 的字段名，用下面的命令查看全部可扫字段：

```bash
python -m xrf_sim.experiments.scripts.scan_params --list-params
```

### 可扫描字段（常用）

| 字段 | 说明 |
|------|------|
| `energy_eV` | 光子能量 (eV) |
| `d_spacing` | 晶面间距 (m) |
| `detector_distance` | wafer → slit 距离 (mm) |
| `detector_theta_deg` | 探测器掠射角 (deg) |
| `crystal_distance` | 晶体到探测器距离 (mm) |
| `fwhm_deg` | 反射率 FWHM (deg) |
| `spot_radius` | 光斑半径 (mm) |
| `ROC` | 弯晶曲率半径 (mm) |
| `a_crystal` | slit 到晶体距离 (mm) |
| `N_photons` | 光子总数 |
| `cone_angle_deg` | 发射锥半角 (deg) |
| `pinhole_radius` | 针孔半径 (mm) |
| `is_flatted` | Soller 狭缝开关 (True/False) |

> `detector_size` / `crystal_size` / `pc_window_size` 是元组，**不能**直接扫描；
> 需要时请改 `--det_size` / `--cryst_size` / `--pc_size` 基础参数。

### 扫描专用参数

| 参数 | 说明 |
|------|------|
| `--sweep "NAME=..."` | 扫描规则，可重复；`NAME=v1,v2,...` 或 `NAME=START:STOP:NUM` |
| `--out_dir` | CSV / PNG 输出目录（默认 `sim_output/scans/`） |
| `--tag` | 输出文件名后缀，便于区分多组实验 |
| `--N` | 每次模拟的光子数（默认 200,000；正式结论建议 2,000,000+） |
| `--seed` | 随机种子，给定后每次模拟可复现 |
| `--profile` / `--profile_x_range` / `--profile_y_range` | 强度分布文件（同 `debug_tool`） |
| `--region_rect` / `--region_circle` | 额外输出各区域的 `pc_frac[...]` / `n_pc[...]` 列 |
| `--no_plot` | 只写 CSV，不画图 |
| `--list-params` | 列出所有可扫描字段后退出 |

### 输出

- `sim_output/scans/scan_<crystal>_<params>.csv`：每次模拟一行，**追加写入**
- `sim_output/scans/scan_<crystal>_<params>.png`：
  一维为 4 联折线图（晶体命中 / 反射 / PC 命中 / PC 效率）；
  二维为 `N_pc_hit` 热图（格内标注数值）；三维以上不画图，只出 CSV

汇总示例：

```bash
# 结果直接读回分析
python -c "import pandas as pd; d=pd.read_csv('sim_output/scans/scan_spiral_fwhm_deg.csv'); print(d.to_string())"
```

单元格里跑（等价写法）：

```python
import numpy as np
from xrf_sim.simulate import simulate, SimulationConfig
from xrf_sim.experiments.scanner import run_scan

def single_run(fwhm):
    r = simulate(SimulationConfig(crystal_type="spiral", a_crystal=160,
                                  fwhm_deg=fwhm, N_photons=200_000, verbose=False))
    return {"fwhm_deg": fwhm, "N_reflected": r.N_reflected,
            "N_pc_hit": r.N_pc_hit, "pc_efficiency": r.pc_efficiency}

df = run_scan("fwhm_deg", np.linspace(0.01, 0.2, 10), single_run,
              csv_path="sim_output/scans/scan_spiral_fwhm.csv")
```

### 与 `experiments/scripts/` 里已有脚本的关系

| 脚本 | 扫描对象 | 是否含 PC 窗口 |
|------|---------|---------------|
| `scan_flat_angle.py` | 平晶 `detector_theta_deg` 25→45 | ❌ 只到反射光子 |
| `scan_curved_roc.py` | 弯晶 `crystal_ROC` 80→300 | ❌ 只到反射光子 |
| `scan_curved_distance.py` | 弯晶 `detector_distance` 80→200 | ❌ 只到反射光子 |
| **`scan_params.py`** | **任意 `SimulationConfig` 字段（1~2 维）** | ✅ 含 PC 命中 |

老脚本是硬编码参数的实验记录，保留作参考；新的扫描统一用 `scan_params.py`。
