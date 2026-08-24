/// Franka Hand timing probe that sweeps Move speed.
//
// Same free-running reader as csv_gripper_timing_probe, but the commanded speed changes every
// trial and there is NO rest between moves -- each Move is issued the instant the previous one
// returns. Two questions:
//
//   * is the command latency (send -> first motion) constant, or does it scale with speed?
//   * is the tail (last motion -> move() returning) constant, or does it scale?
//
// A latency that is constant in seconds is a fixed cost to plan around; one that scales with the
// stroke is really part of the motion. Back-to-back moves also remove the idle rest, so nothing
// here can be blamed on the hand having gone quiet.
//
// Writes samples and commands as two CSVs; analysis lives in Python. MOVES THE FINGERS.

#include <franka/exception.h>
#include <franka/gripper.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;

auto GRIPPER_OPEN_M = 0.07;
auto GRIPPER_CLOSED_M = 0.02;
auto GRIPPER_SETUP_SPEED_M_P_S = 0.05;

// Commanded speeds, one per trial. Spans 20x, so a latency that scales with speed and one that
// does not are impossible to confuse. The top end is past the hand's rated 50 mm/s deliberately:
// if it clips, the fitted speed will say so.
const double SPEEDS_M_P_S[] = {0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10};
const int N_TRIALS = static_cast<int>(sizeof(SPEEDS_M_P_S) / sizeof(SPEEDS_M_P_S[0]));

/// Seconds since the probe started, so every stamp is one comparable double.
const Clock::time_point EPOCH = Clock::now();
double now() { return Seconds(Clock::now() - EPOCH).count(); }

/// One (instant, width) sample off the gripper's own state stream.
struct Sample {
    double t;
    // GripperState.time, which is NOT a clock: libfranka builds it as Duration(message_id), so it
    // is the hand's datagram counter reinterpreted as milliseconds (convertGripperState, libfranka
    // src/gripper.cpp). A gap in it is datagrams we never saw.
    uint64_t message_id;
    double width;
};

struct CommandRecord {
    const char* label;
    double target_width;
    double speed;
    double t_sent;
    double t_complete;
};

// Only the reader thread touches these, and main reads them only after the join -- so the join is
// the synchronisation and no mutex is needed.
std::vector<Sample> samples;
std::atomic<bool> reading{true};

void read_loop(franka::Gripper& gripper) {
    while (reading) {
        try {
            auto state = gripper.readOnce();
            // Stamped on return, the closest instant to the measurement libfranka exposes.
            samples.push_back({now(), state.time.toMSec(), state.width});
        } catch (const franka::Exception& e) {
            // A read that fails while a command is in flight is expected, not fatal.
            std::fprintf(stderr, "  [read] %s\n", e.what());
        }
    }
}

/// One blocking Move, timed. The reader thread keeps sampling straight through it.
CommandRecord move(franka::Gripper& gripper, const char* label, double width, double speed) {
    CommandRecord rec{label, width, speed, now(), 0.0};
    try {
        // A false return is a command the hand DECLINED (an unhomed hand declines every Move) --
        // no exception, and readOnce() keeps working, so nothing else shows it.
        if (!gripper.move(width, speed)) {
            std::fprintf(stderr, "  move REFUSED (returned false)\n");
        }
    } catch (const franka::Exception& e) {
        std::fprintf(stderr, "  move failed: %s\n", e.what());
    }
    rec.t_complete = now();
    return rec;
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        printf("usage: ./speed_gripper_timing_probe <arm_ip> [out_prefix]\n");
        return -1;
    }
    const std::string prefix = (argc == 3) ? argv[2] : "speed_probe";

    franka::Gripper gripper(argv[1]);
    samples.reserve(20000);
    std::vector<CommandRecord> commands;

    // Started before anything is commanded, so every command lands inside a stream already running.
    std::thread reader(read_loop, std::ref(gripper));

    printf("Homing gripper...\n");
    if (!gripper.homing()) {
        printf("Homing failed\n");
        return -1;
    }

    printf("Moving to start position...\n");
    commands.push_back(move(gripper, "start", GRIPPER_CLOSED_M, GRIPPER_SETUP_SPEED_M_P_S));

    for (int i = 0; i < N_TRIALS; i++) {
        // No rest: the next Move goes out the instant this one returns.
        const CommandRecord m = move(gripper, "move",
                                     (i % 2 == 0) ? GRIPPER_OPEN_M : GRIPPER_CLOSED_M,
                                     SPEEDS_M_P_S[i]);
        commands.push_back(m);
        printf("%5.0f mm/s -> %4.0f mm: move() blocked %7.1f ms\n", SPEEDS_M_P_S[i] * 1e3,
               m.target_width * 1e3, (m.t_complete - m.t_sent) * 1e3);
    }

    // Keep reading a moment past the last command so the final settle is in the log too.
    std::this_thread::sleep_for(Seconds(1.0));
    reading = false;
    reader.join();

    FILE* out = fopen((prefix + "_samples.csv").c_str(), "w");
    fprintf(out, "t,message_id,width\n");
    for (const Sample& s : samples) {
        fprintf(out, "%.6f,%llu,%.6f\n", s.t, static_cast<unsigned long long>(s.message_id),
                s.width);
    }
    fclose(out);

    out = fopen((prefix + "_commands.csv").c_str(), "w");
    fprintf(out, "label,target_width,speed,t_sent,t_complete\n");
    for (const CommandRecord& c : commands) {
        fprintf(out, "%s,%.6f,%.6f,%.6f,%.6f\n", c.label, c.target_width, c.speed, c.t_sent,
                c.t_complete);
    }
    fclose(out);

    printf("\nwrote %s_samples.csv (%zu samples) and %s_commands.csv (%zu commands)\n",
           prefix.c_str(), samples.size(), prefix.c_str(), commands.size());
    return 0;
}
