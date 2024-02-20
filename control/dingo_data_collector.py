import rospy
import os
import csv
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from puma_motor_msgs.msg import MultiFeedback


class VelocityMonitor:
    def __init__(self):
        rospy.init_node('data_collector', anonymous=True)

        # Publishers
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Subscribers
        rospy.Subscriber('/joint_states', JointState, self.joint_states_callback)
        rospy.Subscriber('/feedback', MultiFeedback, self.feedback_callback)
        rospy.Subscriber('/-', MultiFeedback, self.pose_callback)

        # File path for data logging
        self.data_folder = os.path.join(os.path.expanduser('~'), 'velocity_data')
        os.makedirs(self.data_folder, exist_ok=True)  # Create the data folder

        self.current_position = None

        self.monitor = False

        self.position = []
        self.velocity = []
        self.acceleration = []

        self.travel = []
        self.current = []
        self.duty_cycle = []
        
    def joint_states_callback(self, data):

        if self.monitor:
            timestamp = data.header.timestamp

            self.position.append([timestamp, *data.position])
            self.velocity.append([timestamp, *data.velocity])
            self.acceleration.append([timestamp, *data.acceleration])

    def feedback_callback(self, data):

        if self.monitor:
            timestamp = data.header.timestamp

            travel = [feedback.travel for feedback in data.drivers_feedback]
            current = [feedback.current for feedback in data.drivers_feedback]
            duty_cycle = [feedback.duty_cycle for feedback in data.drivers_feedback]

            self.travel.append([timestamp,travel])
            self.current.append([timestamp, current])
            self.duty_cycle.append([timestamp, duty_cycle])

    def pose_callback(self, data):
        self.current_position = 0.0

    def move_forward(self):

        velocity = Twist()

        self.monitor = True
        end_position = None

        velocity.linear.x = 0.5
        self.vel_pub.publish(self.velocity)

        while not rospy.is_shutdown():
            if not end_position:
                end_position = self.current_position[0] + 1.0 
            
            if self.current_position[0] >= end_position:
                break

        velocity.linear.x = 0. 
        self.vel_pub.publish(self.velocity)

        self.monitor = False

    def save_data(self, mass):
        # Create a folder with the mass amount as the folder name
        mass_folder = os.path.join(self.data_folder, str(mass))
        os.makedirs(mass_folder, exist_ok=True)

        # Save each data list to a separate CSV file
        data_lists = {
            'position': self.position,
            'velocity': self.velocity,
            'acceleration': self.acceleration,
            'travel': self.travel,
            'current': self.current,
            'duty_cycle': self.duty_cycle
        }

        for data_name, data_list in data_lists.items():
            file_name = f"{data_name}.csv"
            file_path = os.path.join(mass_folder, file_name)

            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Timestamp', *range(len(data_list[0][1]))])  # Write the header

                # Write data to the CSV file
                for timestamp, data in data_list:
                    writer.writerow([timestamp, *data])

        self.position = []
        self.velocity = []
        self.acceleration = []

        self.travel = []
        self.current = []
        self.duty_cycle = []

    def request_mass(self):
        try:
            mass = float(input("Please enter the mass of the pushed object: "))
            return mass
        except ValueError:
            rospy.logerr("Invalid input for mass. Please enter a valid number.")
            return self.request_mass()

    def run(self):
        num_runs = int(input("Enter the number of runs: "))
        for run in range(num_runs):
            mass = self.request_mass()
            # self.move_forward()
            self.save_data(mass)
            rospy.sleep(2)  # Delay between runs


def main():
    vel_monitor = VelocityMonitor()
    vel_monitor.run()

