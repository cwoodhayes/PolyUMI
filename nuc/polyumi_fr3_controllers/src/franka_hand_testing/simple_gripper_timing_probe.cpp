/// minimal timing probe for Franka Hand

#include <franka/exception.h>
#include <franka/gripper.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <future>
#include <thread>
#include <vector>

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

/// Seconds since the probe started, so every stamp is one comparable double.
const Clock::time_point EPOCH = Clock::now();
double now() { return Seconds(Clock::now() - EPOCH).count(); }

/// One (instant, width) sample off the gripper's own state stream.
struct Sample {
  double t;
  // The hand's own stamp for this state, seconds since robot start. Millisecond resolution
  // (Duration is integer ms underneath), and on a different origin from t.
  double franka_t;
  double width;
};

struct CommandRecord {
    double t_sent;
    double t_complete;
    double width;
    // t_sent expressed in the hand's clock, inferred from the first sample that came back.
    double franka_t_sent;
};

struct TrialResult {
    CommandRecord cmd;
    // first sample which we marked as "starting to move"
    Sample first_motion_sample;
    // first sample at which we're within the threshold away from the target position
    Sample last_motion_sample;
};

class GripperWrapper {
public:
    GripperWrapper(char* ip_addr): gripper_(ip_addr) {};

    bool move(double width, double speed) {
        return gripper_.move(width, speed);
    }

    bool homing() {
        return gripper_.homing();
    }

    /// The write thread: it exists only while the move does, and main keeps sampling meanwhile.
    /// move() blocks until the motion ENDS, so timing the call from main would measure the stroke.
    std::future<CommandRecord> move_async(double width, double speed) {
        return std::async(std::launch::async, [this, width, speed] {
            CommandRecord rec{now(), 0.0, width, 0.0};  // franka_t_sent filled in by main
            try {
                // A false return is a command the hand DECLINED (an unhomed hand declines every
                // Move) -- no exception, and readOnce() keeps working, so nothing else shows it.
                if (!gripper_.move(width, speed)) {
                    std::fprintf(stderr, "  move REFUSED (returned false)\n");
                }
            } catch (const franka::Exception& e) {
                std::fprintf(stderr, "  move failed: %s\n", e.what());
            }
            rec.t_complete = now();
            return rec;
        });
    }

    /// @brief blocking read
    /// @return width in m, with timestamps from both PC clock and franka clock
    Sample read_sample() {
        auto state = gripper_.readOnce();
        return {now(), state.time.toSec(), state.width};
    }


private:
    franka::Gripper gripper_;
};


int main(int argc, char** argv) {
    if (argc != 2) {
        printf("usage: ./simple_gripper_timing_probe <arm_ip>\n");
        return -1;
    }

    std::vector<TrialResult> results;
    GripperWrapper gripper(argv[1]);

    // Read-rate accounting, over every trial including any that get discarded.
    std::size_t total_samples = 0;
    std::size_t total_intervals = 0;
    double total_read_span = 0.0;    // wall seconds spent sampling
    double total_franka_span = 0.0;  // the hand's own clock over those same samples

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
        std::vector<Sample> gripper_samples;
        gripper_samples.reserve(MAX_GRIPPER_READS);

        // open the gripper & time it
        auto gripper_speed = GRIPPER_MAX_SPEED_M_P_S;
        auto expected_move_duration = (GRIPPER_OPEN_M - GRIPPER_CLOSED_M) / gripper_speed;
        printf("--- Opening gripper... ");
        auto move_future = gripper.move_async(GRIPPER_OPEN_M, gripper_speed);

        // collect samples for the expected move time + 1s
        auto run_time = expected_move_duration + 1.0;
        auto complete_time = now() + run_time;
        while (now() < complete_time && gripper_samples.size() < MAX_GRIPPER_READS) {
            gripper_samples.push_back(gripper.read_sample());
        }

        // wait for the actual call to finish; nominally it already has.
        auto cmd_record = move_future.get();

        if (gripper_samples.empty()) {
            printf("no state arrived -- DISCARDED\n");
            continue;
        }

        // calculate when the command was sent in the franka clock, by grabbing the franka timestamp
        // of the first read, and then subtracting the PC wall clock time elapsed between the command send
        // and that first read. 
        // NOTE -- IT TURNS OUT franka's State.time is actually not a timestamp at all; it is a datagram
        // sequence number. this is mislabeled in libfranka. so this math isn't actually helpful.
        const Sample& first_sample = gripper_samples.front();
        cmd_record.franka_t_sent = first_sample.franka_t - (first_sample.t - cmd_record.t_sent);

        total_samples += gripper_samples.size();
        total_intervals += gripper_samples.size() - 1;
        total_read_span += gripper_samples.back().t - first_sample.t;
        total_franka_span += gripper_samples.back().franka_t - first_sample.franka_t;

        // using the gripper samples we took just now,
        // find when the gripper started moving using the threshold
        auto rest = gripper_samples.front().width;
        auto first = std::find_if(gripper_samples.begin(), gripper_samples.end(),
            [rest](const Sample& s) {
                return std::fabs(s.width - rest) > GRIPPER_MOTION_THRESHOLD_M;
            });
        auto last = std::find_if(gripper_samples.begin(), gripper_samples.end(),
            [](const Sample& s) {
                return std::fabs(s.width - GRIPPER_OPEN_M) <= GRIPPER_MOTION_THRESHOLD_M;
            });
        if (first == gripper_samples.end() || last == gripper_samples.end()) {
            printf("no motion detected over %zu samples -- DISCARDED\n", gripper_samples.size());
        } else {
            printf("onset %.1f ms (%.1f ms by franka clock), motion %.3f s (%.1f ms franka), blocked %.1f ms\n", (first->t - cmd_record.t_sent) * 1e3,
                   (first->franka_t - cmd_record.franka_t_sent) * 1e3, last->t - first->t,
                   (last->franka_t - first->franka_t),
                   (cmd_record.t_complete - cmd_record.t_sent) * 1e3);
            results.push_back({cmd_record, *first, *last});
        }

        // back to the start, so every trial times an opening from rest
        gripper.move(GRIPPER_CLOSED_M, GRIPPER_MAX_SPEED_M_P_S);
        // rest for a bit to make sure there's nothing funky going on with commanding back to back
        std::this_thread::sleep_for(Seconds(.5));
    }

    if (results.empty()) {
        printf("\nNo usable trials.\n");
        return -1;
    }

    // print out a final report of useful numbers.
    double lat_sum = 0.0, lat_min = 1e9, lat_max = 0.0, blocked_sum = 0.0, motion_sum = 0.0;
    double settle_sum = 0.0;
    for (const auto& r : results) {
        auto latency = r.first_motion_sample.t - r.cmd.t_sent;
        lat_sum += latency;
        lat_min = std::min(lat_min, latency);
        lat_max = std::max(lat_max, latency);
        blocked_sum += r.cmd.t_complete - r.cmd.t_sent;
        motion_sum += r.last_motion_sample.t - r.first_motion_sample.t;
        settle_sum += r.cmd.t_complete - r.last_motion_sample.t;
    }
    auto n = static_cast<double>(results.size());
    printf("\n%zu/%d trials\n", results.size(), N_TRIALS);
    printf("command -> first motion: mean %.1f ms, min %.1f ms, max %.1f ms\n",
           lat_sum / n * 1e3, lat_min * 1e3, lat_max * 1e3);
    printf("move() blocked:          mean %.1f ms\n", blocked_sum / n * 1e3);
    printf("motion duration:         mean %.3f s (commanded %.3f s at %.0f mm/s)\n",
           motion_sum / n, (GRIPPER_OPEN_M - GRIPPER_CLOSED_M) / GRIPPER_MAX_SPEED_M_P_S,
           GRIPPER_MAX_SPEED_M_P_S * 1e3);
    // Target reached (within the motion threshold) -> move() complete. 
    printf("motion stop -> return:   mean %.1f ms\n", settle_sum / n * 1e3);
    printf("gripper read rate:       %.1f Hz (%zu samples over %.1f s)\n",
           total_intervals / total_read_span, total_samples, total_read_span);
    printf("hand clock vs PC clock:  %.3f s of GripperState.time per wall second\n",
           total_franka_span / total_read_span);

    return 0;
}
