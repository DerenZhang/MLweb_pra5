import requests
import time
import csv
import matplotlib.pyplot as plt

BASE = "http://serve-sentiment-env.eba-zhbzbm3i.ca-central-1.elasticbeanstalk.com"

test_cases = {
    "real_1": "The government announced a new healthcare reform plan today.",
    "real_2": "Scientists discovered a potential cure for the common cold after years of research.",
    "fake_1": "Aliens landed in Toronto and offered free Wi-Fi to everyone.",
    "fake_2": "Drinking only coffee for a week makes you lose 10 pounds instantly."
}

csv_file = "api_latency_results.csv"

print("Running functional tests...")
for label, text in test_cases.items():
    r = requests.post(
        f"{BASE}/predict",
        headers={"Content-Type": "application/json"},
        json={"message": text},
        timeout=15
    )
    print(f"{label}: {r.status_code}, {r.text}")

print("\nRunning performance tests (100 requests per test case)...")
results = {}

for label, text in test_cases.items():
    latencies = []
    for i in range(100):
        start = time.time()
        r = requests.post(
            f"{BASE}/predict",
            headers={"Content-Type": "application/json"},
            json={"message": text},
            timeout=15
        )
        end = time.time()
        latencies.append((end - start) * 1000)  
    results[label] = latencies

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["test_case", "latency_ms"])
    for label, latencies in results.items():
        for l in latencies:
            writer.writerow([label, l])


plt.figure(figsize=(8, 5))
plt.boxplot(results.values(), labels=results.keys())
plt.ylabel("Latency (ms)")
plt.title("API Latency per Test Case (ms)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("api_latency_boxplot.png")
plt.show()

