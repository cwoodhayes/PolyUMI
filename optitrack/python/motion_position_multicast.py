#3rd party library imports (from Optitrack)
from localization_helper.NatNetClient import NatNetClient

#Native imports
import sys
import time
import numpy as np
import socket  # Import the socket library for networking
import struct

TOTAL_ATTEMPTS = 3
INTEGER_MULTIPLIER = 10000 #we will multiply all our float positions by this number before packing them into integers to send over the network. This allows us to preserve 4 decimal places of precision, which is more than enough for our purposes (hopefully) and keeps our message sizes smaller than if we were to send floats.

#Class used to handle the underlying localization code for the platform.

class Localizer():
    def __init__(self):
        self.reported_id = 0
        self.reported_pos = 0
        self.reported_rot = 0

        self.send_sequence = 0

    def unpack_pos(self, pos):
        self.reported_id = pos[1]
        self.reported_pos = pos[2]
        self.reported_rot = pos[3]

    def begin_process(self):
        
        #initialize options dict which will hold info for the specific connection.
        self.optionsDict = {}
        self.optionsDict["clientAddress"] = "127.0.0.1"#"192.168.18.158" #This is my computer as a client
        #self.optionsDict["clientAddress"] = "192.168.18.142" #this is the raspberry pi as a client.
        self.optionsDict["serverAddress"] = "127.0.0.1"#"192.168.18.189"
        self.optionsDict["use_multicast"] = True
        
        #This will create a new NatNet client
        self.optionsDict = my_parse_args(sys.argv, self.optionsDict)

        #create a new streaming client and initailize its values.
        self.streaming_client = NatNetClient()
        self.streaming_client.set_client_address(self.optionsDict["clientAddress"])
        self.streaming_client.set_server_address(self.optionsDict["serverAddress"])
        self.streaming_client.set_use_multicast(self.optionsDict["use_multicast"])

        # Configure the streaming client to call our rigid body handler on the emulator to send data out.
        self.streaming_client.new_frame_listener = self.receive_new_frame
        self.streaming_client.rigid_body_listener = receive_rigid_body_frame

        # Start up the streaming client now that the callbacks are set up.
        # This will run perpetually, and operate on a separate thread.
        self.is_running = self.streaming_client.initialize_optitrack_comms()
        if not self.is_running:
            print("ERROR: Could not start streaming client.")
            try:
                self.streaming_client.shutdown()
                sys.exit(1)
            except SystemExit:
                print("...")
            finally:
                self.is_running = False
                print("exiting")

        self.is_looping = True
        time.sleep(1)
        

    #The main function for this class. The one that is the target of a process when a process is intitiated.
    def classifier_behavior(self):

        # START = time.perf_counter()
        START = time.time()
        def now():
            # return time.perf_counter() - START
            return time.time() - START

        port = 54321
        multicast_group = '224.1.1.1'
        base_station_ip = '192.168.18.102'
        base_station_port = 30003

        #set up UDP socket.
        #AF_INET => use IPv4, SOCK_DGRAM => use UDP, IPPROTO_UPD => explicitly says use UDP (just to avoid any ambiguity.)
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

        #Set Time-To-Live (TTL) to 1 (restricts packets to the local subnet)
        ttl = struct.pack('b',1)
        server_socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)

        server_socket.bind(('192.168.18.101',0))


        #Reuse the address and the port (so multiple things can bind to it)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        

        # #increase the buffer size - NOT NEEDED ANYMORE!
        # server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 131072)


        #Create another socket for listening to messages from the air traffic controller
        #This creates a UDP socket 
        rec_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        #allow the socket address to be reused
        rec_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #We're going to listen to anything coming over the port
        rec_sock.bind(('', base_station_port))
        #slightly increase the buffer to hanlde burst traffic more smoothly
        rec_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        #make this blocking for 0s (so we never wait for it)
        rec_sock.setblocking(False)
        #make it so we can't hear our own messages
        rec_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 0)

        #variables for holding information that will be used to transmit charging locations and homing locations.
        saved_home_info = []
        homing_message_counter = 1
        homing_message_live = False
        homing_message_attempts = 0
        saved_homing_message = b''

        saved_chargers_info = []
        chargers_message_counter = 1
        chargers_message_live = False
        chargers_message_attempts = 0
        saved_chargers_message = b''


        #begin process by connecting to MOTIVE
        self.begin_process()
        self.streaming_client.set_major(3)
        self.streaming_client.set_minor(0)
        print('process started.')

        #initialize clocks to control when we print data
        time_last_printed = now()
        time_between_prints = 0.01 #s
        last_time = now()
        last_ten_freq = np.zeros(100)

        avg_pointer = 0

        print('foobar')

        target_freq = 100 #in Hz, this is the frequency we want to be running at. We will print our actual frequency every time we print data to see how close we are to this.
        dt = 1/target_freq
        dt = dt/2 #we will actually try to run at 2x the desired frequency to ensure we don't miss any data from MOTIVE, which is running at 100Hz. We will only send data out at the desired frequency though, as close as we can get given processing time.
        dt = 0
        time_of_last_send = time.perf_counter()


        last_time_stamp = 0
        last_time_stamp2 = 0

        print('Entering main loop...')

        try:
            i = 0
            while self.is_looping:

                # if now() > 5:
                #     print('5 seconds have passed, ending process.')
                #     print(self.send_sequence)
                #     break
                
                #get position of drone(s)
                all_pos = self.streaming_client.get_pos() #returns [Timestamp,[True, 1, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0)], [True, 2, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0)], ...]
                time_stamp = np.round(all_pos[0]/100,3) #the first element of the list is the timestamp for that frame, the rest are the positions of the rigid bodies.
                # time_stamp = now()
                # print(time_stamp)

                # print(np.round(abs((time_stamp - last_time_stamp) - (time_stamp2 - last_time_stamp2)),6))
                # p2 = self.streaming_client.get_pos()
                # print(pos[2])
                # pos = [True, 1, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0)]
                # print(time_stamp)

                if time_stamp <= last_time_stamp:
                    print('WARNING: Duplicate or OLD timestamp received, skipping frame...')
                    continue
                last_time_stamp = time_stamp
                # last_time_stamp2 = time_stamp2

                saved_home_info = []
                saved_chargers_info = []

                all_pos = all_pos[1:] #remove the timestamp from the list so we are left with just the positions.
                #If there is some data there, unpack it.
                if len(all_pos) > 0:

                    time_new_pos = now()
                    if time_new_pos - last_time == 0:
                        # print('DIV 0 WARNING')
                        last_time = time_new_pos        
                    else:
                        freq = 1/(time_new_pos - last_time)

                        last_time = time_new_pos

                        #remove outliers
                        if freq < 500:


                            last_ten_freq[avg_pointer] = freq
                            avg_freq = np.average(last_ten_freq)
                            median_freq = np.median(last_ten_freq)
                            avg_pointer += 1
                            if avg_pointer >= 100:
                                avg_pointer = 0

                    message1 = struct.pack('5sfH', b'opti1', time_stamp, self.send_sequence) #4s is the string 'opti', f is float, H is unsigned short int
                    message2 = struct.pack('5sfH', b'opti2', time_stamp, self.send_sequence) #4s is the string 'opti', f is float, H is unsigned short int
                    msg_payload = struct.pack('B', 0) #available charger(s)
                    message1 += msg_payload
                    message2 += msg_payload
                    num_bodies = 0
                    for pos in all_pos:
                        int_form = []
                        if pos[0]:
                       
                            self.unpack_pos(pos) #writes to self.reported_id, self.reported_pos, self.reported_rot (rotation is in quaternion form)

                            if pos[1] > 70 and pos[1] != 100:
                                continue 
                            elif pos[1] == 100:
                                # print('hundo: '+ + str(self.reported_pos) +  str(quaternion_to_euler(self.reported_rot)))
                                continue

                            msg = b''
                            #pos[1] is reported id
                            msg_id = struct.pack('B',pos[1])
                            

                            #UNCOMMENT IF YOU WANT TO SEND INTEGERS
                            # msg += msg_id
                            #pack the x,y,z position.
                            # for item in self.reported_pos:
                            #     integer_form = int(item * 10000)
                            #     msg_payload = struct.pack('h', integer_form)
                            #     msg += msg_payload
                            #     int_form.append(integer_form)
                            # #pack the rotation in quaternion form.
                            # for item in self.reported_rot:
                            #     integer_form = int(item * 10000)
                            #     msg_payload = struct.pack('h', integer_form)
                            #     msg += msg_payload
                            #     int_form.append(integer_form)

                            #Uncomment if you want to send FLOATS
                            msg_payload = struct.pack('7f',pos[2][0],pos[2][1],pos[2][2],self.reported_rot[0],self.reported_rot[1],self.reported_rot[2],self.reported_rot[3])
                            msg = msg_id + msg_payload

                            if pos[1] < 30:
                                message1 += msg
                            else:
                                message2 += msg

                            num_bodies += 1

                            #NO NEED TO SAVE HOMING INFO (DRONE WILL HOME FROM ITS OWN POSITION.)
                            # if pos[1] == 5:
                            #     saved_home_info.append(pos)

                            if pos[1] in [201, 202, 203, 5, 14, 9]: #these are the IDs for the charging station locations. 5, 10, 9 are in there for test only.
                                saved_chargers_info.append(pos)

                        else:
                            print("WARNING: Position not received for id " + str(pos[1]) + ". Check that Motive streaming is on.")

                    
                    # Send the message to the broadcast address
                    # message = "Hello, from optitrack."
                    # encoded_message = message.encode('utf-8')
                    server_socket.sendto(message1, (multicast_group, port))
                    # time.sleep(0.0005)
                    time.sleep(0.001)
                    server_socket.sendto(message2, (multicast_group, port))
                    # time.sleep(0.0005)
                    time.sleep(0.001)
                    
                    self.send_sequence = self.send_sequence + 1
                    #Since we pack the sequence number as a short unsigned integer (2 bytes), we need to prevent an overflow error by resetting it.
                    if self.send_sequence == 65536:
                        self.send_sequence = 0
                        
                    try:
                        if now()> time_last_printed + time_between_prints:
                            # print(pos)
                            # print(int_form)
                            # print(f"Broadcasting message: {message1}")
                            # print(len(message1))
                            # print(num_bodies)

                            print("Loop frequency: %0.2f, %0.2f, %0.2f, %d" %(freq, avg_freq, median_freq, num_bodies)) 

                            # print(np.max(last_ten_freq))
                            # print(pos[2][0], str(pos[2][0])[0:12])
                            # print(last_ten_freq)
                            # print(roll)
                            time_last_printed = now()
                    except:
                        print('error')

                    # time.sleep(2)


                if self.streaming_client.connected() is False:
                    print("ERROR: Could not connect properly.  Check that Motive streaming is on.")
                    try:
                        self.streaming_client.shutdown()
                        sys.exit(2)
                    except SystemExit:
                        print("...")
                    finally:
                        self.is_running = False
                        print("exiting")
                        self.is_looping = False
                # if not pos[0]:
                #     print("ERROR: Position not received. Check that Motive streaming is on.")
                #     try:
                #         self.streaming_client.shutdown()
                #         sys.exit(2)
                #     except SystemExit:
                #         print("...")
                #     finally:
                #         self.is_running = False
                #         print("exiting")
                #         self.is_looping = False


                #check if we got anything from the air traffic controller and address it here.
                try:
                    data,addr = rec_sock.recvfrom(1024)
                    unpacked_data = struct.unpack('6s',data)
                    
                    if unpacked_data[0] == b'homing' and not homing_message_live:
                    
                        #saved_home_info is optitrack information that is saved in the for loop that parses  the optitrack rigid bodies. 
                        #we do not bother building message 4 up there because it takes too long and is inefficient to do it every for loop when we don't need it 99% of the time.
                        #when we do need it, we build it here and then save it so we can retransmit TOTAL_ATTEMPTS times
                        #DEVELOPERS NOTE: I left his code in, but really it never executes because we don't save home info anymore - drone uses its own pos for home.
                        message4 = struct.pack('5sBBB',b'opti4',homing_message_counter,1,42) #the 0/1 is for not calculating/calculating and the 42 is unused
                        # for pos in saved_home_info:
                        #     msg_id = struct.pack('B',pos[1])
                        #     msg_payload = struct.pack('7f',pos[2][0],pos[2][1],pos[2][2],pos[3][0],pos[3][1],pos[3][2],pos[3][3])
                        #     msg = msg_id + msg_payload

                        #     message4 += msg    
                        homing_message_live = True      
                        saved_homing_message = message4      

                    elif unpacked_data[0] == b'charge' and not chargers_message_live:
                        #saved_chargers_info is optitrack information that is saved in the for loop that parses  the optitrack rigid bodies. 
                        #we do not bother building message 3 up there because it takes too long and is inefficient to do it every for loop when we don't need it 99% of the time.
                        #when we do need it, we build it here and then save it so we can retransmit TOTAL_ATTEMPTS times
                        message3 = struct.pack('5sB',b'opti3',chargers_message_counter)
                        for pos in saved_chargers_info:
                            self.unpack_pos(pos) #writes to self.reported_id, self.reported_pos, self.reported_rot (rotation is in quaternion form)

                            msg = b''

                            # uncomment to send floats
                            msg_id = struct.pack('B',pos[1])
                            msg_payload = struct.pack('7f',pos[2][0],pos[2][1],pos[2][2],pos[3][0],pos[3][1],pos[3][2],pos[3][3])
                            msg = msg_id + msg_payload

                            # msg_id = struct.pack('B',pos[1])
                            # msg += msg_id

                            # #UNCOMMENT IF YOU WANT TO SEND INTEGERS
                            # #pack the x,y,z position.
                            # for item in self.reported_pos:
                            #     integer_form = int(item * 10000)
                            #     msg_payload = struct.pack('h', integer_form)
                            #     msg += msg_payload
                            # #pack the rotation in quaternion form.
                            # for item in self.reported_rot:
                            #     integer_form = int(item * 10000)
                            #     msg_payload = struct.pack('h', integer_form)
                            #     msg += msg_payload

                            message3 += msg    
                        chargers_message_live = True      
                        saved_chargers_message = message3
                        print(int_form)

                    

                except BlockingIOError:
                    pass

                #transmit the homing message IFF it is live
                if homing_message_live:
                    server_socket.sendto(saved_homing_message, (multicast_group, port))
                    # print('homing sent...', homing_message_counter, homing_message_attempts)

                    homing_message_attempts += 1

                    #if we have reached our max attempts, we reset everything so we are ready to go for the next homing message.
                    if homing_message_attempts >= TOTAL_ATTEMPTS:
                        homing_message_counter += 1
                        if homing_message_counter == 256:
                            homing_message_counter = 0
                        homing_message_live = False
                        homing_message_attempts = 0

                #transmit the charging message IFF it is live
                if chargers_message_live:
                    server_socket.sendto(saved_chargers_message, (multicast_group, port))
                    # print('chargers sent...', chargers_message_counter, chargers_message_attempts)

                    chargers_message_attempts += 1

                    #if we have reached our max attempts, we reset everything so we are ready to go for the next chargers message.
                    if chargers_message_attempts >= TOTAL_ATTEMPTS:
                        chargers_message_counter += 1
                        if chargers_message_counter == 256:
                            chargers_message_counter = 0
                        chargers_message_live = False
                        chargers_message_attempts = 0

                
        except KeyboardInterrupt:
            print("Server stopped.")
        finally:
            # Close the socket when done
            rec_sock.close()
            server_socket.close()
            print('Socket closed.')
            
        


    
    # This is a callback function that gets connected to the NatNet client
    # and called once per mocap frame.
    def receive_new_frame(data_dict):
        order_list=[ "frameNumber", "markerSetCount", "unlabeledMarkersCount", "rigidBodyCount", "skeletonCount",
                    "labeledMarkerCount", "timecode", "timecodeSub", "timestamp", "isRecording", "trackedModelsChanged" ]
        dump_args = False
        if dump_args == True:
            out_string = "    "
            for key in data_dict:
                out_string += key + "="
                if key in data_dict :
                    out_string += data_dict[key] + " "
                out_string+="/"
            print(out_string)

# This is a callback function that gets connected to the NatNet client. It is called once per rigid body per frame
def receive_rigid_body_frame( new_id, position, rotation ):
    pass
    #print( "Received frame for rigid body", new_id )
    #print( "Received frame for rigid body", new_id," ",position," ",rotation )




#from PythonSample.py in optitrack natnet installation file.
def my_parse_args(arg_list, args_dict):
    # set up base values
    arg_list_len=len(arg_list)
    if arg_list_len>1:
        args_dict["serverAddress"] = arg_list[1]
        if arg_list_len>2:
            args_dict["clientAddress"] = arg_list[2]
        if arg_list_len>3:
            if len(arg_list[3]):
                args_dict["use_multicast"] = True
                if arg_list[3][0].upper() == "U":
                    args_dict["use_multicast"] = False

    return args_dict


def quaternion_to_euler(q):
    # Extract quaternion components
    w, x, y, z = q

    # Convert quaternion to rotation matrix
    rotation_matrix = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])

    # Extract roll, pitch, and yaw from rotation matrix
    # Roll (x-axis rotation)
    roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

    # Pitch (y-axis rotation)
    sin_pitch = rotation_matrix[2, 0]
    cos_pitch = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    pitch = np.arctan2(sin_pitch, cos_pitch)

    # Yaw (z-axis rotation)
    sin_yaw = rotation_matrix[1, 0]
    cos_yaw = rotation_matrix[0, 0]
    yaw = np.arctan2(sin_yaw, cos_yaw)

    # # Convert angles from radians to degrees
    # roll = np.degrees(roll)
    # pitch = np.degrees(pitch)
    # yaw = np.degrees(yaw)

    return roll, pitch, yaw


if __name__ == "__main__":

    localizer = Localizer()
    localizer.classifier_behavior()
    

