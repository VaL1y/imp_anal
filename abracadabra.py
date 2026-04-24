import random
import matplotlib.pyplot as plt
import sqlite3

# параметры
seed = 42
noise_sigma_ms = 0
num_impulses_in_pack = 15
generation_duration_s = 15.0
num_packs_to_plot = 3
min_interval_ms = 10
max_interval_ms = 100
epsilon_s = 0.005
tolerance_ms = 1.0

db_name = "experiment_results"

clear_db_before_run = False


def init_database(db_name):
    db_file = f"{db_name}.db"
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS results 
                 (sigma REAL, seed INTEGER, found_period INTEGER, result TEXT)''')
    conn.commit()
    conn.close()


def clear_database(db_name):
    db_file = f"{db_name}.db"
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute('DROP TABLE IF EXISTS results')
    conn.commit()
    conn.close()
    print(f"База данных {db_file} очищена.")


def save_to_db(db_name, sigma, seed_val, found_period, result_text):
    db_file = f"{db_name}.db"
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("INSERT INTO results VALUES (?, ?, ?, ?)",
              (round(sigma, 4), seed_val, found_period if found_period is not None else -1, result_text))
    conn.commit()
    conn.close()


def get_statistics(db_name):
    db_file = f"{db_name}.db"
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("""SELECT sigma, 
                        COUNT(*) as total,
                        SUM(CASE WHEN result='правильный' THEN 1 ELSE 0 END) as correct,
                        SUM(CASE WHEN result='неправильный' THEN 1 ELSE 0 END) as wrong,
                        SUM(CASE WHEN result='не найден' THEN 1 ELSE 0 END) as not_found
                 FROM results 
                 GROUP BY sigma 
                 ORDER BY sigma""")
    stats = c.fetchall()
    conn.close()
    return stats

def plot_results(stats, db_name):
    if not stats:
        print("Нет данных для графика статистики.")
        return

    sigmas_list = [row[0] for row in stats]
    perc_correct = [row[2] / row[1] * 100 for row in stats]
    perc_wrong = [row[3] / row[1] * 100 for row in stats]
    perc_not_found = [row[4] / row[1] * 100 for row in stats]

    plt.figure(figsize=(12, 6))
    plt.plot(sigmas_list, perc_correct, 'g-', linewidth=2, label='Правильный период')
    plt.plot(sigmas_list, perc_wrong, 'r-', linewidth=2, label='Неправильный период')
    plt.plot(sigmas_list, perc_not_found, 'b--', linewidth=2, label='Не найден')
    plt.xlabel('Дисперсия шума (мс)')
    plt.ylabel('Процент экспериментов (%)')
    plt.title(f'Устойчивость анализатора')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------

init_database(db_name)

if clear_db_before_run:
    clear_database(db_name)
    init_database(db_name)
    print("База данных очищена перед запуском.\n")

print(
    f"ТЕКУЩИЙ ЗАПУСК → БД = {db_name}.db | σ = {noise_sigma_ms:.4f} мс | seed = {seed} | tolerance = {tolerance_ms} мс\n")

random.seed(seed)

# ------ пачки
impulse_rel_times = [0]
for _ in range(num_impulses_in_pack - 1):
    delta = random.randint(min_interval_ms, max_interval_ms)
    impulse_rel_times.append(impulse_rel_times[-1] + delta)

first_interval_ms = impulse_rel_times[1]
inter_pack_distance_ms = first_interval_ms

# ----- дисперсия
all_times = []
current_pack_start = 0.0
max_time = generation_duration_s * 1000

while current_pack_start <= max_time:
    for rel_time in impulse_rel_times:
        noise = random.gauss(0, noise_sigma_ms)
        actual_time = current_pack_start + rel_time + noise
        all_times.append(actual_time)
    pack_end = current_pack_start + impulse_rel_times[-1]
    current_pack_start = pack_end + inter_pack_distance_ms

# -----
with open('impulse_times.csv', 'w', encoding='utf-8') as f:
    for t in all_times:
        f.write(f"{t:.4f}\n")

# -------
times_ms = []
with open('impulse_times.csv', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            times_ms.append(float(line.strip()))

found_period = None
for i in range(2, len(times_ms)):
    upper_sum = times_ms[i - 1] - times_ms[0]
    lower_sum = times_ms[i] - times_ms[1]
    if abs(upper_sum - lower_sum) <= tolerance_ms:
        found_period = i
        break

# -----
if found_period is None:
    result_text = "не найден"
    print("Период НЕ НАЙДЕН")
elif found_period == num_impulses_in_pack:
    result_text = "правильный"
    print(f"Период найден ПРАВИЛЬНО — {found_period} импульсов")
else:
    result_text = "неправильный"
    print(f"Период найден НЕПРАВИЛЬНО — получено {found_period}, ожидалось {num_impulses_in_pack}")

# ------
save_to_db(db_name, noise_sigma_ms, seed, found_period, result_text)
print(f"Результат сохранён в БД: {db_name}.db")

# ---- график 1
impulses_to_plot = num_packs_to_plot * num_impulses_in_pack
plot_times_s = [t / 1000.0 for t in all_times[:impulses_to_plot]]

separator_times = []
for i in range(num_packs_to_plot):
    last_idx = (i + 1) * num_impulses_in_pack - 1
    if last_idx < len(plot_times_s):
        sep_time = plot_times_s[last_idx] + epsilon_s
        separator_times.append(sep_time)

plt.figure(figsize=(14, 5))
plt.vlines(plot_times_s, 0, 1, colors='blue', linewidth=1.5, label='Импульсы')
plt.vlines(separator_times, 0, 1, colors='red', linewidth=1.5, linestyles='--', label='Границы пачек')
plt.xlabel('Время (секунды)')
plt.ylabel('Амплитуда импульса')
plt.title(f'Последовательность импульсов (первые {num_packs_to_plot} пачек) | σ = {noise_sigma_ms:.4f} мс')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ------ график 2
stats = get_statistics(db_name)
if stats:
    plot_results(stats, db_name)
else:
    print("В выбранной базе пока нет данных.")
