/// Franka Hand timing probe with a free-running reader thread.
//
// Same experiment as simple_gripper_timing_probe -- 10 open/close trials, same widths and speed --
// but one thread does nothing except readOnce() for the WHOLE session, not just while a Move is
// outstanding. So the width history is continuous across settling and idle gaps, and main can
// issue plain blocking moves instead of the std::async the per-trial version needed.
//
// Writes samples and commands as two CSVs; analysis lives in Python, not here. MOVES THE FINGERS.

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

#define N_TRIALS 10

auto GRIPPER_OPEN_M = 0.07;
auto GRIPPER_CLOSED_M = 0.02;
auto GRIPPER_MAX_SPEED_M_P_S = 0.05;

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
CommandRecord move(franka::Gripper& gripper, const char* label, double width) {
    CommandRecord rec{label, width, now(), 0.0};
    try {
        // A false return is a command the hand DECLINED (an unhomed hand declines every Move) --
        // no exception, and readOnce() keeps working, so nothing else shows it.
        if (!gripper.move(width, GRIPPER_MAX_SPEED_M_P_S)) {
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
        printf("usage: ./csv_gripper_timing_probe <arm_ip> [out_prefix]\n");
        return -1;
    }
    const std::string prefix = (argc == 3) ? argv[2] : "gripper_timing";

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
    commands.push_back(move(gripper, "start", GRIPPER_CLOSED_M));

    for (int i = 0; i < N_TRIALS; i++) {
        commands.push_back(move(gripper, "open", GRIPPER_OPEN_M));
        // back to the start, so every trial times an opening from rest
        commands.push_back(move(gripper, "close", GRIPPER_CLOSED_M));
        // rest for a bit to make sure there's nothing funky going on with commanding back to back
        std::this_thread::sleep_for(Seconds(.5));
        const CommandRecord& open = commands[commands.size() - 2];
        printf("trial %d: move() blocked %.1f ms\n", i, (open.t_complete - open.t_sent) * 1e3);
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
    fprintf(out, "label,target_width,t_sent,t_complete\n");
    for (const CommandRecord& c : commands) {
        fprintf(out, "%s,%.6f,%.6f,%.6f\n", c.label, c.target_width, c.t_sent, c.t_complete);
    }
    fclose(out);

    printf("\nwrote %s_samples.csv (%zu samples) and %s_commands.csv (%zu commands)\n",
           prefix.c_str(), samples.size(), prefix.c_str(), commands.size());
    return 0;
}
