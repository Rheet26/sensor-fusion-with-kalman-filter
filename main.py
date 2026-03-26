import random
import numpy as np
import matplotlib.pyplot as plt
random.seed(42)
np.random.seed(42)

# ---------------- CONFIG ----------------
LOW = 10
HIGH = 90
DEFAULT = 50
ITER = 60

SCENARIOS = ["normal", "spike", "drift", "mixed"]


# ---------------- SENSOR ----------------
class Sensor:
    def __init__(self):
        self.prev = None
        self.health = 100

    def read(self, t, scenario):
        base = 50

        if scenario == "spike" and 20 < t < 25:
            return random.randint(95, 110)

        if scenario == "drift":
            base += t * 0.3

        if scenario == "mixed":
            r = random.random()
            if r < 0.1:
                return random.randint(0, 5)
            elif r < 0.2:
                return random.randint(95, 100)
            elif r < 0.3:
                return 42

        return int(np.random.normal(base, 8))


# ---------------- FAULT DETECTION ----------------
def detect_fault(value, prev):
    if value < LOW or value > HIGH:
        return "range"
    if prev is not None and value == prev:
        return "stuck"
    if prev is not None and abs(value - prev) > 30:
        return "spike"
    return None


# ---------------- FUSION ----------------
def weighted_fusion(values, sensors):
    valid = []
    weights = []

    for v, s in zip(values, sensors):
        if LOW <= v <= HIGH:
            valid.append(v)
            weights.append(max(s.health, 1))

    if not valid:
        return None

    return np.average(valid, weights=weights)


# ---------------- ADAPTIVE KALMAN FILTER ----------------
class AdaptiveKalmanFilter:
    def __init__(self):
        self.x = 50          # IMPORTANT: start near true value
        self.P = 1
        self.Q = 0.05       # smaller process noise (more stable)
        self.R = 8          # slightly higher base measurement noise
        self.alpha = 0.05   # VERY SMALL adaptation rate

    def update(self, measurement):
        # prediction
        self.P += self.Q

        # innovation (error)
        innovation = abs(measurement - self.x)

        # VERY controlled adaptation
        self.R = (1 - self.alpha) * self.R + self.alpha * min(max(innovation, 1), 15)

        # gain
        K = self.P / (self.P + self.R)

        # update
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * self.P

        return self.x

# ---------------- SENSOR SIMULATION ----------------
def run_scenario(scenario):
    sensors = [Sensor(), Sensor(), Sensor()]
    kf = AdaptiveKalmanFilter()

    last_valid = None
    gt, fusion, kalman, raw = [], [], [], []

    faults = 0
    total = 0

    for t in range(ITER):
        values = [s.read(t, scenario) for s in sensors]
        raw.append(values[0])   # store first sensor as raw baseline
        true_val = 50 + (t * 0.3 if scenario == "drift" else 0)
        gt.append(true_val)

        for i, s in enumerate(sensors):
            fault = detect_fault(values[i], s.prev)

            if fault:
                s.health -= 20
                faults += 1
            else:
                s.health = min(100, s.health + 5)

            s.prev = values[i]

        out = weighted_fusion(values, sensors)

        if out is None:
            out = last_valid if last_valid is not None else DEFAULT
        else:
            last_valid = out

        fusion.append(out)
        kalman.append(kf.update(out))

        total += len(values)

    gt = np.array(gt)
    fusion = np.array(fusion)
    kalman = np.array(kalman)

    rmse_fusion = np.sqrt(np.mean((gt - fusion) ** 2))
    rmse_kalman = np.sqrt(np.mean((gt - kalman) ** 2))
    improvement = ((rmse_fusion - rmse_kalman) / rmse_fusion) * 100

    print(f"\n--- {scenario.upper()} ---")
    print(f"Fault Rate: {round((faults/total)*100,2)}%")
    print(f"RMSE (Fusion): {round(rmse_fusion,2)}")
    print(f"RMSE (Kalman): {round(rmse_kalman,2)}")
    print(f"Improvement: {round(rmse_fusion,2)} → {round(rmse_kalman,2)}")
    print(f"Improvement %: {improvement:.2f}")

    return gt, fusion, kalman, raw


# ---------------- DRONE SIMULATION ----------------
def drone_simulation():
    steps = 60

    true_x, true_y = [], []
    x, y = 0, 0

    for _ in range(steps):
        x += 1
        y += 0.5
        true_x.append(x)
        true_y.append(y)

    noisy_x = [val + np.random.normal(0, 2) for val in true_x]
    noisy_y = [val + np.random.normal(0, 2) for val in true_y]

    # 2D Kalman (position + velocity)
    def kalman_2d(noisy):
        x = np.array([[noisy[0]], [0]])
        P = np.eye(2)

        F = np.array([[1, 1], [0, 1]])
        H = np.array([[1, 0]])

        Q = np.array([[0.1, 0], [0, 0.1]])
        R = np.array([[4]])

        results = []

        for z in noisy:
            x = F @ x
            P = F @ P @ F.T + Q

            y = np.array([[z]]) - (H @ x)
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)

            x = x + K @ y
            P = (np.eye(2) - K @ H) @ P

            results.append(x[0, 0])

        return results

    filtered_x = kalman_2d(noisy_x)
    filtered_y = kalman_2d(noisy_y)

    # RMSE
    true_x_arr = np.array(true_x)
    true_y_arr = np.array(true_y)
    noisy_x_arr = np.array(noisy_x)
    noisy_y_arr = np.array(noisy_y)
    filt_x_arr = np.array(filtered_x)
    filt_y_arr = np.array(filtered_y)

    rmse_noisy = np.sqrt(np.mean((true_x_arr - noisy_x_arr)**2 + (true_y_arr - noisy_y_arr)**2))
    rmse_filtered = np.sqrt(np.mean((true_x_arr - filt_x_arr)**2 + (true_y_arr - filt_y_arr)**2))

    print("\n--- DRONE NAVIGATION SIMULATION ---")
    print(f"RMSE (Noisy GPS): {rmse_noisy:.2f}")
    print(f"RMSE (Kalman): {rmse_filtered:.2f}")
    print(f"Improvement: {((rmse_noisy - rmse_filtered)/rmse_noisy)*100:.2f}%")

    return true_x, true_y, noisy_x, noisy_y, filtered_x, filtered_y


# ---------------- MAIN ----------------
def main():
    print("\nRunning Sensor Fusion Scenarios...\n")

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Sensor plots
    for i, scenario in enumerate(SCENARIOS):
        gt, fusion, kalman, raw = run_scenario(scenario)

        axes[i].plot(gt, label="Ground Truth")
        axes[i].plot(raw, label="Raw Sensor", linestyle='dashed')
        axes[i].plot(fusion, label="Fusion")
        axes[i].plot(kalman, label="Kalman")

        axes[i].set_title(f"{scenario.capitalize()} Scenario")
        axes[i].legend()

    print("\nRunning Drone Simulation...\n")

    tx, ty, nx, ny, fx, fy = drone_simulation()

    axes[4].plot(tx, ty, label="True Path")
    axes[4].plot(nx, ny, label="Noisy GPS")
    axes[4].plot(fx, fy, label="Kalman Path")

    axes[4].set_title("Drone Navigation")
    axes[4].legend()

    axes[5].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()