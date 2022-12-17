import logging
from collections import deque, namedtuple
from pathlib import Path

import numpy as np

Transition = namedtuple(
    "Transition", ["state", "action", "next_state", "reward", "done"]
)


class ReplayBuffer:
    def __init__(self, max_capacity=5e6):
        self.logger = logging.getLogger(__name__)
        self.unsaved_transitions = 0
        self.curr_file_idx = 1
        self.replay_buffer = deque(maxlen=int(max_capacity))

    def __len__(self) -> int:
        return len(self.replay_buffer)

    def clear(self):
        self.replay_buffer.clear()

    def add_transition(self, state, action, next_state, reward, done):
        """
        This method adds a transition to the replay buffer.
        """
        transition = Transition(state, action, next_state, reward, done)
        self.replay_buffer.append(transition)
        self.unsaved_transitions += 1

    def sample(self, batch_size: int):
        indices = np.random.choice(
            len(self.replay_buffer),
            min(len(self.replay_buffer), batch_size),
            replace=False,
        )
        states, actions, next_states, rewards, dones = zip(
            *[self.replay_buffer[idx] for idx in indices]
        )
        return (
            np.array(states),
            np.array(actions),
            np.array(next_states),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
        )

    def save(self, path="./replay_buffer"):
        if path is None:
            return False
        if self.unsaved_transitions > 0:
            p = Path(path)
            p = p.expanduser()
            p.mkdir(parents=True, exist_ok=True)
            final_rb_index = len(self.replay_buffer)
            start_rb_index = len(self.replay_buffer) - self.unsaved_transitions
            for replay_buffer_index in range(start_rb_index, final_rb_index):
                transition = self.replay_buffer[replay_buffer_index]
                file_name = "%s/transition_%09d.npz" % (path, self.curr_file_idx)
                np.savez(
                    file_name,
                    state=transition.state,
                    action=transition.action,
                    next_state=transition.next_state,
                    reward=transition.reward,
                    done=transition.done,
                )
                self.curr_file_idx += 1
            # Logging
            if self.unsaved_transitions == 1:
                self.logger.info(
                    "Saved file with index : %09d" % (self.curr_file_idx - 1)
                )
            else:
                self.logger.info(
                    "Saved files with indices : %09d - %09d"
                    % (
                        self.curr_file_idx - self.unsaved_transitions,
                        self.curr_file_idx - 1,
                    )
                )
            self.unsaved_transitions = 0
            return True
        return False

    def load(self, path="./replay_buffer"):
        if path is None:
            return False
        p = Path(path)
        p = p.expanduser()
        if p.is_dir():
            p = p.glob("*.npz")
            files = [x for x in p if x.is_file()]
            self.curr_file_idx = len(files) + 1
            files = files[: self.replay_buffer.maxlen]
            if len(files) > 0:
                for file in files:
                    data = np.load(file, allow_pickle=True)
                    transition = Transition(
                        data["state"].item(),
                        data["action"],
                        data["next_state"].item(),
                        data["reward"].item(),
                        data["done"].item(),
                    )
                    self.replay_buffer.append(transition)
                self.logger.info(
                    "Replay buffer loaded successfully %d files" % len(files)
                )
                return True
            else:
                self.logger.info("No files were found in path %s" % (path))
        else:
            self.logger.info("Path %s is not a directory" % (path))
        return False
