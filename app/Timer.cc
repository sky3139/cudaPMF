
#include "Timer.hpp"
#include <iostream>
#include <string>
#include <iomanip>
#define WIDTH 20
#define Print std::cout << std::setw(WIDTH)

using namespace std::chrono;

Timer::Timer() : started_(false), paused_(false) {}
Timer::Timer(std::string msg) : msg(msg) { Start(); }

void Timer::Start()
{
    started_ = true;
    paused_ = false;
    start_time_ = high_resolution_clock::now();
}

void Timer::Restart()
{
    started_ = false;
    Start();
}

void Timer::Pause()
{
    paused_ = true;
    pause_time_ = high_resolution_clock::now();
}

void Timer::Resume()
{
    paused_ = false;
    start_time_ += high_resolution_clock::now() - pause_time_;
}

void Timer::Reset()
{
    started_ = false;
    paused_ = false;
}

double Timer::ElapsedMicroSeconds() const //微秒
{
    if (!started_)
    {
        return 0.0;
    }
    if (paused_)
    {
        return duration_cast<microseconds>(pause_time_ - start_time_).count();
    }
    else
    {
        return duration_cast<microseconds>(high_resolution_clock::now() -
                                           start_time_)
            .count();
    }
}

double Timer::ElapsedSeconds() const { return ElapsedMicroSeconds() / 1e6; }

double Timer::ElapsedMinutes() const { return ElapsedSeconds() / 60; }

double Timer::ElapsedHours() const { return ElapsedMinutes() / 60; }

void Timer::PrintSeconds(const std::string &str) const
{
    std::cout.setf(std::ios::left); //设置对齐方式为left
    Print << str << ElapsedSeconds() << " s"
          << "\n";
}

void Timer::PrintMinutes() const
{
    printf("Elapsed time: %.3f [minutes]\n", ElapsedMinutes());
}

void Timer::PrintHours() const
{
    printf("Elapsed time: %.3f [hours]\n", ElapsedHours());
}
