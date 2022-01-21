/*
 * File:   Log.h
 * Author: Alberto Lepe <dev@alepe.com>
 *
 * Created on December 1, 2015, 6:00 PM
 */

// Adapted by Philipp Lindenberger (Phil26AT)

#pragma once
#include <ctime>
#include <iostream>
#include <mutex>
#include <third-party/progressbar.h>
//

namespace pixsfm {

// https://stackoverflow.com/questions/16357999/current-date-and-time-as-string
inline std::string GetCurrentDateTimeString() {
  time_t rawtime;
  struct tm* timeinfo;
  char buffer[80];

  time(&rawtime);
  timeinfo = localtime(&rawtime);

  strftime(buffer, sizeof(buffer), "%Y/%m/%d %H:%M:%S", timeinfo);
  std::string str(buffer);
  return str;
}

enum headless { CERR = 2, COUT };

enum typelog {
  DEBUG = 0,  // CERR
  INFO,       // COUT
  WARN,       // CERR
  ERROR
};

struct structlog {
  bool silence_normal = false;
  bool headers = true;
  int level = 0;
  std::mutex mutex;
};

extern structlog LOGCFG;

class STDLOG {
 public:
  STDLOG() {}
  STDLOG(headless type) {
    msglevel = int(type);
    opened = false;
  }
  STDLOG(typelog type) {
    complete_endl = true;
    msglevel = type;
    if (LOGCFG.headers && !LOGCFG.silence_normal) {
      operator<<("[" + GetCurrentDateTimeString() + " " + getLabel(type) +
                 "] ");
    }
  }
  ~STDLOG() {
    // if(opened && complete_endl) {
    //     getOStream() << std::endl;
    // }
    // opened = false;
  }
  template <class T>
  STDLOG& operator<<(const T& msg) {
    if (msglevel >= LOGCFG.level && !LOGCFG.silence_normal) {
      getOStream() << msg;
      opened = true;
    }
    return *this;
  }

  STDLOG& operator<<(std::ostream& (*os)(std::ostream&)) {
    if (msglevel >= LOGCFG.level && !LOGCFG.silence_normal) {
      getOStream() << os;
      opened = true;
    }
    return *this;
  }

 protected:
  bool opened = false;
  bool complete_endl = false;
  int msglevel = int(DEBUG);
  inline std::string getLabel(typelog type) {
    std::string label = "pixsfm ";
    switch (type) {
      case DEBUG:
        label += "DEBUG";
        break;
      case INFO:
        label += "INFO";
        break;
      case WARN:
        label += "WARNING";
        break;
      case ERROR:
        label += "ERROR";
        break;
    }
    return label;
  }
  std::ostream& getOStream() {
    return (msglevel == 1 || msglevel == 3) ? std::cout : std::cerr;
  }
};

class SYNC_LOG : public STDLOG {
 public:
  SYNC_LOG() { LOGCFG.mutex.lock(); }
  SYNC_LOG(typelog type) {
    msglevel = int(type);
    LOGCFG.mutex.lock();
    if (LOGCFG.headers) {
      operator<<("[" + GetCurrentDateTimeString() + " " + getLabel(type) +
                 "] ");
    }
  }
  SYNC_LOG(headless type) {
    complete_endl = true;
    opened = false;
    msglevel = int(type);
    LOGCFG.mutex.lock();
  }
  ~SYNC_LOG() {
    // if(opened && complete_endl) {
    //     getOStream() << std::endl;
    // }
    // opened = false;
    LOGCFG.mutex.unlock();
  }
  template <class T>
  SYNC_LOG& operator<<(const T& msg) {
    if (msglevel >= LOGCFG.level) {
      getOStream() << msg;
      opened = true;
    }
    return *this;
  }
  SYNC_LOG& operator<<(std::ostream& (*os)(std::ostream&)) {
    if (msglevel >= LOGCFG.level) {
      getOStream() << os;
      opened = true;
    }
    return *this;
  }
};

class LogProgressbar : public progressbar {
 public:
  LogProgressbar() : progressbar() {}
  LogProgressbar(size_t n, bool showbar = true, bool init_update = true)
      : progressbar(n, showbar, false) {
    if (init_update) update();
  }
  inline void update(size_t inc = 1) override {
    if (update_is_called) progress += inc;
    std::string out = get_update_str();
    if (out.c_str() != "\r") {
      STDLOG(CERR) << out;
    }
  }
};

class SyncLogProgressbar : public progressbar {
 public:
  SyncLogProgressbar() : progressbar() {}
  SyncLogProgressbar(size_t n, bool showbar = true, bool init_update = true)
      : progressbar(n, showbar, false) {
    if (init_update) update();
  }
  inline void update(size_t inc = 1) override {
    // std::lock_guard<std::mutex> lock(mutex_);
    if (update_is_called) progress += inc;
    std::string out = get_update_str();
    if (out.c_str() != "\r") {
      SYNC_LOG(CERR) << out;
    }
  }
};

}  // namespace pixsfm
