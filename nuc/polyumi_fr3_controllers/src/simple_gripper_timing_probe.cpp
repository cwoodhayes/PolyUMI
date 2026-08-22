/// minimal timing probe for Franka Hand

#include <franka/exception.h>
#include <franka/gripper.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>
#include <future>

using Clock = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;

#define MAX_GRIPPER_READS 5000
#define N_TRIALS 10

auto GRIPPER_OPEN_M = 0.07;
auto GRIPPER_CLOSED_M = 0.02;
auto GRIPPER_MAX_SPEED_M_P_S = 0.05;

// if the gripper moves more than this from the initial
// position we say it's moving
auto GRIPPER_MOTION_THRESHOLD_M = 0.001;

/// One (instant, width) sample off the gripper's own state stream.
struct Sample {
  double t;
  double width;
};

struct CommandRecord {
    double t_sent;
    double t_complete;
    double width;
};

class GripperWrapper {
public:
    GripperWrapper(char* ip_addr): gripper_(franka::Gripper(ip_addr)) {};

    bool move(double width, double speed) {
        gripper_.move(width, speed);
    }

    bool homing() {
        gripper_.homing();
    }

    std::future<CommandRecord> move_async(double width, double speed) {
        // signal with the semaphore
    }

    double read_width() {
        auto state = gripper_.readOnce();
        return state.width;
    }


private:

    void write_loop() {
        // wait for signal from main thread (e.g., acquire binary semaphore, use condition variable/mutex
        // record the "started timestamp"
        // initiate the action

    };

    franka::Gripper gripper_;
};


int main(int argc, char** argv) {
    std::vector<Sample> gripper_samples;
    std::vector<CommandRecord> gripper_commands;
    gripper_samples.reserve(MAX_GRIPPER_READS);
    gripper_commands.reserve(2 * N_TRIALS);

    GripperWrapper gripper(argv[1]);
    Clock wall_time;

    if (argc != 1) {
        printf("usage: ./simple_gripper_timing_probe <arm_ip>\n");
    }

    printf("Homing gripper...\n");
    auto res = gripper.homing();
    if (!res) {
        printf("Homing failed");
        return -1;
    }

    printf("Moving to start position...\n");
    gripper.move(GRIPPER_CLOSED_M, GRIPPER_MAX_SPEED_M_P_S);

    for (int i=0; i<N_TRIALS; i++) {
        printf("\nBEGIN Trial %d\n\n", i);

        // open the gripper & time it
        auto gripper_speed = GRIPPER_MAX_SPEED_M_P_S;
        auto expected_move_duration = (GRIPPER_OPEN_M - GRIPPER_CLOSED_M) / gripper_speed;
        printf("--- Opening gripper... ");
        auto move_future = gripper.move_async(GRIPPER_OPEN_M, gripper_speed);

        // collect samples for the expected move time + 1s
        auto run_time = expected_move_duration + 1.0;
        auto complete_time = Seconds(run_time) + wall_time.now();
        while (wall_time.now() < complete_time) {
            // TODO sample the gripper width into 
        }

        // wait for the actual call to finish; nominally it already has.
        auto cmd_record = move_future.get();

        // using the gripper samples we took just now, 
        // find when the gripper started moving using the threshold
        // TODO
        
    }

    return 0;
}