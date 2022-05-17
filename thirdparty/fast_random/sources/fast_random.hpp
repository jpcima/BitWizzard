// SPDX-License-Identifier: BSD-2-Clause

#pragma once
#include <type_traits>
#include <limits>
#include <array>
#include <cmath>
#include <cstdint>

/**
 * @brief A low-quality random number generator guaranteed to be very fast
 */
class fast_rand {
public:
    typedef uint32_t result_type;

    fast_rand() noexcept
    {
    }

    explicit fast_rand(uint32_t value) noexcept
        : mem(value)
    {
    }

    static constexpr uint32_t min() noexcept
    {
        return 0;
    }

    static constexpr uint32_t max() noexcept
    {
        return std::numeric_limits<uint32_t>::max();
    }

    uint32_t operator()() noexcept
    {
        uint32_t next = mem * 1664525u + 1013904223u; // Numerical Recipes
        mem = next;
        return next;
    }

    void seed(uint32_t value = 0) noexcept
    {
        mem = value;
    }

    void discard(unsigned long long z) noexcept
    {
        for (unsigned long long i = 0; i < z; ++i)
            operator()();
    }

private:
    uint32_t mem = 0;
};

/**
 * @brief A uniform real distribution guaranteed to be very fast
 */
template <class T>
class fast_real_distribution {
public:
    static_assert(std::is_floating_point<T>::value, "The type must be floating point.");

    typedef T result_type;

    fast_real_distribution(T a, T b)
        : a_(a), b_(b), k_(b - a)
    {
    }

    template <class G>
    T operator()(G& g) const
    {
        return a_ + (g() - T(G::min())) * (k_ / T(G::max() - G::min()));
    }

    T a() const noexcept
    {
        return a_;
    }

    T b() const noexcept
    {
        return b_;
    }

    T min() const noexcept
    {
        return a_;
    }

    T max() const noexcept
    {
        return b_;
    }

private:
    T a_;
    T b_;
    T k_;
};

/**
 * @brief Global random singletons
 *
 * TODO: could be moved into a singleton class holder
 *
 */
namespace Random {
static fast_rand randomGenerator;
} // namespace Random

/**
 * @brief Generate normally distributed noise.
 *
 * This sums the output of N uniform random generators.
 * The higher the N, the better is the approximation of a normal distribution.
 */
template <class T, unsigned N = 4>
class fast_gaussian_generator {
    static_assert(N > 1, "Invalid quality setting");

public:
    explicit fast_gaussian_generator(float mean, float variance, uint32_t initialSeed = Random::randomGenerator())
    {
        mean_ = mean;
        gain_ = variance / std::sqrt(N / 3.0);
        seed(initialSeed);
    }

    void seed(uint32_t s)
    {
        seeds_[0] = s;
        for (unsigned i = 1; i < N; ++i) {
            s += s * 1664525u + 1013904223u;
            seeds_[i] = s;
        }
    }

    float operator()() noexcept
    {
        float sum = 0;
        for (unsigned i = 0; i < N; ++i) {
            uint32_t next = seeds_[i] * 1664525u + 1013904223u;
            seeds_[i] = next;
            sum += static_cast<int32_t>(next) * (1.0f / (1ll << 31));
        }
        return mean_ + gain_ * sum;
    }

private:
    std::array<uint32_t, N> seeds_ {{}};
    float mean_ { 0 };
    float gain_ { 0 };
};
