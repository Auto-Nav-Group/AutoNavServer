from map import Map
import os
import sys

if sys.platform == "win32":
    ASSET_PATH = "G:\\Projects\\AutoNav\\AutoNavServer\\assets\\drl\\generate_map_files"
elif sys.platform == "linux" or sys.platform == "linux2":
    ASSET_PATH = '/home/jovyan/workspace/AutoNavServer/assets/drl/generate_map_files'
elif sys.platform == "darwin":
    ASSET_PATH = "/Users/maximkudryashov/Projects/AutoNav/AutoNavServer/assets/drl/generate_map_files"
else:
    print("SYSTEM NOT SUPPORTED. EXITING")
    exit()
#ASSET_PATH_ENV = ASSET_PATH + "\\environment"
#ASSET_PATH_DYNAMIC = ASSET_PATH + "\\dynamic"
#ASSET_PATH = "/Users/maximkudryashov/Projects/AutoNav/AutoNavServer/assets/drl/generate_map_files"

def from_map(map):
    for file in os.listdir(ASSET_PATH):
        if file.endswith(".urdf"):
            os.remove(os.path.join(ASSET_PATH, file))

    for file in os.listdir(ASSET_PATH):
        if file.endswith(".urdf"):
            os.remove(os.path.join(ASSET_PATH, file))

    count = 1
    for obstacle in map.obstacles:
        o = open(os.path.join(ASSET_PATH, "obs_" + str(count) + ".urdf"), "w")
        x = obstacle.Loc.x+obstacle.Size.width/2- map.size.width/2
        y = obstacle.Loc.y + obstacle.Size.height/2 - map.size.height/2
        o.write(
            "<?xml version=\"1.0\"?>\n"
            "<robot name=\"obs_" + str(count) + "\">\n"
            "  <link name=\"obs_" + str(count) + "\">\n"
            "    <inertial>\n"
            "      <mass value=\"0\"/>\n"
            "      <inertia ixx=\"0\" ixy=\"0\" ixz=\"0\" iyy=\"0\" iyz=\"0\" izz=\"0\"/>\n"
            "    </inertial>\n"
            "    <collision>\n"
            "      <origin rpy=\"0 0 0\" xyz=\"" + str(x) + " " + str(y) + " 0\"/>\n"
            "      <geometry>\n"
            "        <box size=\"" + str(obstacle.Size.width) + " " + str(obstacle.Size.height) + " 5\"/>\n"                                     
            "      </geometry>\n"
            "    </collision>\n"
            "    <visual>\n"
            "      <origin rpy=\"0 0 0\" xyz=\"" + str(x) + " " + str(y) + " 0\"/>\n"
            "      <geometry>\n"
            "        <box size=\"" + str(obstacle.Size.width) + " " + str(obstacle.Size.height) + " 5\"/>\n"
            "      </geometry>\n"
            "      <material name=\"red\">\n"
            "        <color rgba=\"1 0 0 1\"/>\n"
            "      </material>\n"
            "    </visual>\n"
            "  </link>\n"
            "</robot>\n"
        )
        count += 1
        o.close()

    # Generate the floor URDF
    o = open(os.path.join(ASSET_PATH, "floor.urdf"), "w")
    o.write(
        "<?xml version=\"1.0\"?>\n"
        "<robot name=\"floor\">\n"
        "  <link name=\"floor\">\n"
        "    <inertial>\n"
        "      <mass value=\"0\"/>\n"
        "      <inertia ixx=\"0\" ixy=\"0\" ixz=\"0\" iyy=\"0\" iyz=\"0\" izz=\"0\"/>\n"
        "    </inertial>\n"
        "    <collision>\n"
        "      <origin rpy=\"0 0 0\" xyz=\"0 0 -0.5\"/>\n"
        "      <geometry>\n"
        "        <box size=\"" + str(map.size.width) + " " + str(map.size.height) + " 1\"/>\n"
        "      </geometry>\n"
        "    </collision>\n"
        "    <visual>\n"
        "      <origin rpy=\"0 0 0\" xyz=\"0 0 -0.5\"/>\n"
        "      <geometry>\n"
        "        <box size=\"" + str(map.size.width) + " " + str(map.size.height) + " 1\"/>\n"
        "      </geometry>\n"
        "      <material name=\"white\">\n"
        "        <color rgba=\"1 1 1 1\"/>\n"
        "      </material>\n"
        "    </visual>\n"
        "  </link>\n"
        "</robot>\n"
    )

    # Generate bounds
    #region Bounds
    o = open(os.path.join(ASSET_PATH, "bound1.urdf"), "w")
    o.write(
        "<?xml version=\"1.0\"?>\n"
        "<robot name=\"bound2\">\n"
        "  <link name=\"bound2\">\n"
        "    <inertial>\n"
        "      <mass value=\"0\"/>\n"
        "      <inertia ixx=\"0\" ixy=\"0\" ixz=\"0\" iyy=\"0\" iyz=\"0\" izz=\"0\"/>\n"
        "    </inertial>\n"
        "    <collision>\n"
        "      <origin rpy=\"0 0 0\" xyz=\"0 " + str(map.size.height/2) + " 0\"/>\n"
        "      <geometry>\n"
        "        <box size=\"" + str(map.size.width) + " 0 10\"/>\n"
        "      </geometry>\n"
        "    </collision>\n"
        "    <visual>\n"
        "      <origin rpy=\"0 0 0\" xyz=\"0 " + str(map.size.height/2) + " 0\"/>\n"
        "      <geometry>\n"
        "        <box size=\"" + str(map.size.width) + " 0 10\"/>\n"
        "      </geometry>\n"
        "      <material name=\"red\">\n"
        "        <color rgba=\"1 0 0 0\"/>\n"
        "      </material>\n"
        "    </visual>\n"
        "  </link>\n"
        "</robot>\n"
    )

    o = open(os.path.join(ASSET_PATH, "bound2.urdf"), "w")
    o.write(
        "<?xml version=\"1.0\"?>\n"
        "<robot name=\"bound2\">\n"
        "  <link name=\"bound2\">\n"
        "    <inertial>\n"
        "      <mass value=\"0\"/>\n"
        "      <inertia ixx=\"0\" ixy=\"0\" ixz=\"0\" iyy=\"0\" iyz=\"0\" izz=\"0\"/>\n"
        "    </inertial>\n"
        "    <collision>\n"
        "      <origin rpy=\"0 0 0\" xyz=\"0 -" + str(map.size.height/2) + " 0\"/>\n"
        "      <geometry>\n"
        "        <box size=\"" + str(map.size.width) + " 0 10\"/>\n"
        "      </geometry>\n"
        "    </collision>\n"
        "    <visual>\n"
        "      <origin rpy=\"0 0 0\" xyz=\"0 -" + str(map.size.height/2) + " 0\"/>\n"
        "      <geometry>\n"
        "        <box size=\"" + str(map.size.width) + " 0 10\"/>\n"
        "      </geometry>\n"
        "      <material name=\"red\">\n"
        "        <color rgba=\"1 0 0 0\"/>\n"
        "      </material>\n"
        "    </visual>\n"
        "  </link>\n"
        "</robot>\n"
    )

    o = open(os.path.join(ASSET_PATH, "bound3.urdf"), "w")
    o.write(
        "<?xml version=\"1.0\"?>\n"
        "<robot name=\"bound3\">\n"
        "  <link name=\"bound3\">\n"
        "    <inertial>\n"
        "      <mass value=\"0\"/>\n"
        "      <inertia ixx=\"0\" ixy=\"0\" ixz=\"0\" iyy=\"0\" iyz=\"0\" izz=\"0\"/>\n"
        "    </inertial>\n"
        "    <collision>\n"
        "      <origin rpy=\"0 0 0\" xyz=\"" + str(map.size.width/2) + " 0 0\"/>\n"
        "      <geometry>\n"
        "        <box size=\"0 " + str(map.size.height) + " 10\"/>\n"
        "      </geometry>\n"
        "    </collision>\n"
        "    <visual>\n"
        "      <origin rpy=\"0 0 0\" xyz=\"" + str(map.size.width/2) + " 0 0\"/>\n"
        "      <geometry>\n"
        "        <box size=\"0 " + str(map.size.height) + " 10\"/>\n"
        "      </geometry>\n"
        "      <material name=\"red\">\n"
        "        <color rgba=\"1 0 0 0\"/>\n"
        "      </material>\n"
        "    </visual>\n"
        "  </link>\n"
        "</robot>\n"
    )

    o = open(os.path.join(ASSET_PATH, "bound4.urdf"), "w")
    o.write(
        "<?xml version=\"1.0\"?>\n"
        "<robot name=\"bound4\">\n"
        "  <link name=\"bound4\">\n"
        "    <inertial>\n"
        "      <mass value=\"0\"/>\n"
        "      <inertia ixx=\"0\" ixy=\"0\" ixz=\"0\" iyy=\"0\" iyz=\"0\" izz=\"0\"/>\n"
        "    </inertial>\n"
        "    <collision>\n"
        "      <origin rpy=\"0 0 0\" xyz=\"-" + str(map.size.width/2) + " 0 0\"/>\n"
        "      <geometry>\n"
        "        <box size=\"0 " + str(map.size.height) + " 10\"/>\n"
        "      </geometry>\n"
        "    </collision>\n"
        "    <visual>\n"
        "      <origin rpy=\"0 0 0\" xyz=\"-" + str(map.size.width/2) + " 0 0\"/>\n"
        "      <geometry>\n"
        "        <box size=\"0 " + str(map.size.height) + " 10\"/>\n"
        "      </geometry>\n"
        "      <material name=\"red\">\n"
        "        <color rgba=\"1 0 0 0\"/>\n"
        "      </material>\n"
        "    </visual>\n"
        "  </link>\n"
        "</robot>\n"
    )
    o = open(os.path.join(ASSET_PATH, "robot.urdf"), "w")
    o.write(
        "<?xml version=\"1.0\"?>\n"
        "<robot name=\"robot\">\n"
        "  <link name=\"robot\">\n"
        "    <inertial>\n"
        "      <mass value=\"1\"/>\n"
        "      <inertia ixx=\"0.1\" ixy=\"0\" ixz=\"0\" iyy=\"0.1\" iyz=\"0\" izz=\"0.1\"/>\n"
        "    </inertial>\n"
        "    <collision>\n"
        "      <origin rpy=\"0 0 0\" xyz=\"0 0 0\"/>\n"
        "      <geometry>\n"
        "        <box size=\"1 1 0.5\"/>\n"
        "      </geometry>\n"
        "    </collision>\n"
        "    <visual>\n"
        "      <origin rpy=\"0 0 0\" xyz=\"0 0 0\"/>\n"
        "      <geometry>\n"
        "        <box size=\"1 1 0.5\"/>\n"
        "      </geometry>\n"
        "      <material name=\"blue\">\n"
        "        <color rgba=\"0 0 1 1\"/>\n"
        "      </material>\n"
        "    </visual>\n"
        "  </link>\n"
        "</robot>\n"

    )
    o = open(os.path.join(ASSET_PATH, "goal.urdf"), "w")
    o.write(
        "<?xml version=\"1.0\"?>\n"
        "<robot name=\"goal\">\n"
        "  <link name=\"goal\">\n"
        "    <inertial>\n"
        "      <mass value=\"0\"/>\n"
        "      <inertia ixx=\"0.1\" ixy=\"0\" ixz=\"0\" iyy=\"0.1\" iyz=\"0\" izz=\"0.1\"/>\n"
        "    </inertial>\n"
        "    <visual>\n"
        "      <origin rpy=\"0 0 0\" xyz=\"0 0 0\"/>\n"
        "      <geometry>\n"
        "        <box size=\"1 1 1\"/>\n"
        "      </geometry>\n"
        "      <material name=\"green\">\n"
        "        <color rgba=\"0 1 0 0.5\"/>\n"
        "      </material>\n"
        "    </visual>\n"
        "  </link>\n"
        "</robot>\n"

    )
    #endregion