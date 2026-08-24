/// Franka Hand probe: where does the commanded speed stop being honoured?
//
// Commands speeds far past anything the hand can deliver (up to 800 mm/s) and reads the answer
// off move()'s BLOCKING DURATION rather than the width trace. That matters: above ~150 mm/s a
// full stroke lasts fewer than three 5 Hz width refreshes, so the width can no longer measure a
// speed -- but move() returning is timed to the microsecond. If the hand saturates at some
// xdot_max, blocked time stops falling as the commanded speed rises, and the knee is the answer.
//
// Each speed is run at TWO stroke lengths (60 mm and 30 mm). With one stroke length a term that
// scales with distance and one that depends only on speed are indistinguishable; two lengths
// separate them, which is what the return-time model needs.
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

// Wider than the other probes (60 mm rather than 50) to buy width refreshes at speed, but still
// clear of both hard stops -- the hand homes to about 80 mm.
auto GRIPPER_LOW_M = 0.015;
auto GRIPPER_MID_M = 0.045;
auto GRIPPER_HIGH_M = 0.075;
auto GRIPPER_SETUP_SPEED_M_P_S = 0.05;

// Well past the hand's rated 50 mm/s. 100 mm/s is already known to be honoured, so the knee -- if
// there is one -- lies somewhere above it.
const double SPEEDS_M_P_S[] = {0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80};
const int N_SPEEDS = static_cast<int>(sizeof(SPEEDS_M_P_S) / sizeof(SPEEDS_M_P_S[0]));

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
    const char* outcome;
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

/// One blocking Move, timed. A speed the hand rejects outright shows up as "refused".
CommandRecord move(franka::Gripper& gripper, const char* label, double width, double speed) {
    CommandRecord rec{label, width, speed, now(), 0.0, "ok"};
    try {
        if (!gripper.move(width, speed)) {
            rec.outcome = "refused";
        }
    } catch (const franka::CommandException&) {
        rec.outcome = "aborted";
    } catch (const franka::Exception& e) {
        rec.outcome = "failed";
        std::fprintf(stderr, "  move failed: %s\n", e.what());
    }
    rec.t_complete = now();
    return rec;
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        printf("usage: ./maxspeed_gripper_timing_probe <arm_ip> [out_prefix]\n");
        return -1;
    }
    const std::string prefix = (argc == 3) ? argv[2] : "maxspeed";

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
    commands.push_back(move(gripper, "start", GRIPPER_LOW_M, GRIPPER_SETUP_SPEED_M_P_S));

    for (int i = 0; i < N_SPEEDS; i++) {
        const double v = SPEEDS_M_P_S[i];
        // 60, 30, 30, 60 mm, ending back at LOW ready for the next speed.
        const CommandRecord full_up = move(gripper, "move", GRIPPER_HIGH_M, v);
        commands.push_back(full_up);
        commands.push_back(move(gripper, "move", GRIPPER_MID_M, v));
        commands.push_back(move(gripper, "move", GRIPPER_HIGH_M, v));
        const CommandRecord full_down = move(gripper, "move", GRIPPER_LOW_M, v);
        commands.push_back(full_down);

        printf("%5.0f mm/s: 60 mm strokes blocked %7.1f / %7.1f ms  (%s, %s)\n", v * 1e3,
               (full_up.t_complete - full_up.t_sent) * 1e3,
               (full_down.t_complete - full_down.t_sent) * 1e3, full_up.outcome,
               full_down.outcome);
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
    fprintf(out, "label,target_width,speed,t_sent,t_complete,outcome\n");
    for (const CommandRecord& c : commands) {
        fprintf(out, "%s,%.6f,%.6f,%.6f,%.6f,%s\n", c.label, c.target_width, c.speed, c.t_sent,
                c.t_complete, c.outcome);
    }
    fclose(out);

    printf("\nwrote %s_samples.csv (%zu samples) and %s_commands.csv (%zu commands)\n",
           prefix.c_str(), samples.size(), prefix.c_str(), commands.size());
    return 0;
}
