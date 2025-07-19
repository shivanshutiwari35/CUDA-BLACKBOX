#include <iostream>
#include <vector>
#include <random>
#include <chrono>

int main() {
    const int n = 100000;
    std::vector<double> x(n), y(n);

    // Generate synthetic data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::normal_distribution<> noise(0.0, 0.1);

    for (int i = 0; i < n; ++i) {
        x[i] = dis(gen);
        y[i] = 3 * x[i] + 4 + noise(gen);
    }

    double m = 0, b = 0;
    double lr = 0.01;
    int epochs = 10;

    auto start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double dm = 0, db = 0;
        for (int i = 0; i < n; ++i) {
            double y_pred = m * x[i] + b;
            double error = y[i] - y_pred;
            dm += -2 * x[i] * error;
            db += -2 * error;
        }
        m -= (dm / n) * lr;
        b -= (db / n) * lr;
        std::cout << "Epoch " << epoch + 1 << ": m = " << m << ", b = " << b << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "\nExecution Time: " << duration.count() << " ms\n";

    return 0;
}
