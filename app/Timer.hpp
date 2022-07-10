#pragma once

#include <chrono>
#include <string>
#include <sys/time.h>
class Timer
{
public:
  Timer();
  Timer(std::string msg);
  void Start();
  void Restart();
  void Pause();
  void Resume();
  void Reset();

  double ElapsedMicroSeconds() const;
  double ElapsedSeconds() const;
  double ElapsedMinutes() const;
  double ElapsedHours() const;

  void PrintSeconds(const std::string &str) const;
  void PrintMinutes() const;
  void PrintHours() const;

  static __time_t GetUTC()
  {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    // stringstream s;
    // s<<tv.tv_sec;
    // printf("second:%ld \n", tv.tv_sec);                                 //秒
    // printf("millisecond:%ld \n", tv.tv_sec * 1000 + tv.tv_usec / 1000); //毫秒
    // printf("microsecond:%ld \n", tv.tv_sec * 1000000 + tv.tv_usec);     //微秒
    return tv.tv_sec * 1000000 + tv.tv_usec;
  }
  ~Timer()
  {
    double dt = ElapsedMicroSeconds() * 0.001;
    printf("%s:%.3f ms\n", msg.c_str(), dt);
  }

private:
  bool started_;
  bool paused_;
  std::string msg;
  std::chrono::high_resolution_clock::time_point start_time_;
  std::chrono::high_resolution_clock::time_point pause_time_;
};
