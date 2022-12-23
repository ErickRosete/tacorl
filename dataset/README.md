# Dataset
## Real World
For the real world robot, we collected 9 hours of play data by asking users to teleoperate a Franka Emika
Panda robot arm on a tabletop environment without a specific task in mind.
This environment consists of a table with a drawer that can be opened and closed. It also contains a
sliding door on top of a wooden base, such that the handle can be reached by the end effector. 
On top of the drawer, there are three LED buttons with green, blue and orange coatings. 
When the LED button is clicked, it toggles the state of the light. 
Additionally, there are three different colored blocks with letters on each side.

### Download
To download the real-world dataset please access the following webpage
https://www.kaggle.com/datasets/oiermees/taco-robot

### Data Structure
Each interaction timestep is stored in a dictionary inside a numpy file and contains all corresponding sensory observations, different action spaces, state information and language annoations.

#### Camera Observations
The keys to access the different camera observations are:
```
['rgb_static'] (dtype=np.uint8, shape=(150, 200, 3)),
['rgb_gripper'] (dtype=np.uint8, shape=(200, 200, 3)),
['depth_static'] (dtype=np.float32, shape=(150, 200)),
['depth_gripper'] (dtype=np.float32, shape=(200, 200)),
```
The static RGB-D image was recorded from the Azure Kinect camera and the gripper RGB-D image
from the FRAMOS Industrial Depth Camera D435e.

#### Actions
Actions are in cartesian space and define the desired tcp pose wrt to the world frame and the binary gripper action.
The keys to access the 7-DOF absolute and relative actions are:
(tcp = tool center point, i.e. a virtual frame between the gripper finger tips of the robot)
```
['actions']
(dtype=np.float32, shape=(7,))
tcp position (3): x,y,z in absolute world coordinates
tcp orientation (3): euler angles x,y,z in absolute world coordinates
gripper_action (1): binary (close = -1, open = 1)

['rel_actions_world']
(dtype=np.float32, shape=(7,))
tcp position (3): x,y,z in relative world coordinates normalized and clipped to (-1, 1) with scaling factor 50
tcp orientation (3): euler angles x,y,z in relative world coordinates normalized and clipped to (-1, 1) with scaling factor 20
gripper_action (1): binary (close = -1, open = 1)

['rel_actions_gripper']
(dtype=np.float32, shape=(7,))
tcp position (3): x,y,z in relative gripper frame coordinates normalized and clipped to (-1, 1) with scaling factor 50
tcp orientation (3): euler angles x,y,z in relative gripper frame coordinates normalized and clipped to (-1, 1) with scaling factor 20
gripper_action (1): binary (close = -1, open = 1)
```