#Copyright © 2018 Naturalpoint
#
#Licensed under the Apache License, Version 2.0 (the "License")
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

# OptiTrack NatNet direct depacketization library for Python 3.x

import sys
import socket
import struct
from threading import Thread
import copy
import time
from . import DataDescriptions
from . import MoCapData



def get_message_id(data):
    message_id = int.from_bytes( data[0:2], byteorder='little',  signed=True )
    return message_id


# Create structs for reading various object types to speed up parsing.
Vector2 = struct.Struct( '<ff' )
Vector3 = struct.Struct( '<fff' )
Quaternion = struct.Struct( '<ffff' )
FloatValue = struct.Struct( '<f' )
DoubleValue = struct.Struct( '<d' )
NNIntValue = struct.Struct( '<I')
FPCalMatrixRow = struct.Struct( '<ffffffffffff' )
FPCorners      = struct.Struct( '<ffffffffffff')

class NatNetClient:
    # print_level = 0 off
    # print_level = 1 on
    # print_level = >1 on / print every nth mocap frame
    print_level = 20
    
    def __init__( self ):
        # Change this value to the IP address of the NatNet server.
        self.server_ip_address = "127.0.0.1"

        # Change this value to the IP address of your local network interface
        self.local_ip_address = "127.0.0.1"

        # This should match the multicast address listed in Motive's streaming settings.
        self.multicast_address = "239.255.42.99"

        # NatNet Command channel
        self.command_port = 1510

        # NatNet Data channel
        self.data_port = 1511

        self.use_multicast = True

        # Set this to a callback method of your choice to receive per-rigid-body data at each frame.
        self.rigid_body_listener = None
        self.new_frame_listener  = None

        # Set Application Name
        self.__application_name = "Not Set"

        # NatNet stream version server is capable of. This will be updated during initialization only.
        self.__nat_net_stream_version_server = [0,0,0,0]

        # NatNet stream version. This will be updated to the actual version the server is using during runtime.
        self.__nat_net_requested_version = [0,0,0,0]

        # server stream version. This will be updated to the actual version the server is using during initialization.
        self.__server_version = [0,0,0,0]

        # Lock values once run is called
        self.__is_locked = False

        # Server has the ability to change bitstream version
        self.__can_change_bitstream_version = False

        self.command_thread = None
        self.data_thread = None
        self.command_socket = None
        self.data_socket = None

        self.stop_threads=False

        self.preset_major = False
        self.preset_minor = False


    # Client/server message ids
    NAT_CONNECT               = 0
    NAT_SERVERINFO            = 1
    NAT_REQUEST               = 2
    NAT_RESPONSE              = 3
    NAT_REQUEST_MODELDEF      = 4
    NAT_MODELDEF              = 5
    NAT_REQUEST_FRAMEOFDATA   = 6
    NAT_FRAMEOFDATA           = 7
    NAT_MESSAGESTRING         = 8
    NAT_DISCONNECT            = 9
    NAT_KEEPALIVE             = 10
    NAT_UNRECOGNIZED_REQUEST  = 100
    NAT_UNDEFINED             = 999999.9999


    def set_client_address(self, local_ip_address):
        if not self.__is_locked:
            self.local_ip_address = local_ip_address

    def get_client_address(self):
        return self.local_ip_address

    def set_server_address(self,server_ip_address):
        if not self.__is_locked:
            self.server_ip_address = server_ip_address

    def get_server_address(self):
        return self.server_ip_address


    def set_use_multicast(self, use_multicast):
        if not self.__is_locked:
            self.use_multicast = use_multicast

    def can_change_bitstream_version(self):
        return self.__can_change_bitstream_version

    def set_nat_net_version(self, major, minor):
        """checks to see if stream version can change, then changes it with position reset"""
        return_code = -1
        if self.__can_change_bitstream_version and \
            ((major != self.__nat_net_requested_version[0]) or\
             (minor != self.__nat_net_requested_version[1])):
            sz_command = "Bitstream,%1.1d.%1.1d"%(major, minor)
            return_code = self.send_command(sz_command)
            if return_code >=0:
                self.__nat_net_requested_version[0] = major
                self.__nat_net_requested_version[1] = minor
                self.__nat_net_requested_version[2] = 0
                self.__nat_net_requested_version[3] = 0
                print("changing bitstream MAIN")
                # get original output state
                #print_results = self.get_print_results()
                #turn off output
                #self.set_print_results(False)
                # force frame send and play reset
                self.send_command("TimelinePlay")
                time.sleep(0.1)
                tmpCommands=["TimelinePlay",
                    "TimelineStop",
                    "SetPlaybackCurrentFrame,0",
                    "TimelineStop"]
                self.send_commands(tmpCommands,False)
                time.sleep(2)
                #reset to original output state
                #self.set_print_results(print_results)
            else:
                print("Bitstream change request failed")
        return return_code


    def get_major(self):
        return self.__nat_net_requested_version[0]
    
    def set_major(self, major):
        self.preset_major = major
        return

    def get_minor(self):
        return self.__nat_net_requested_version[1]
    
    def set_minor(self, minor):
        self.preset_minor = minor
        return

    def set_print_level(self, print_level=0):
        if(print_level >=0):
            self.print_level = print_level
        return self.print_level

    def get_print_level(self):
        return self.print_level


    def connected(self):
        ret_value = True
        # check sockets
        if self.data_socket ==None:
            ret_value = False
        return ret_value

    # Create a data socket to attach to the NatNet stream
    def __create_data_socket( self, port ):
        result = None

        if self.use_multicast:
            # Multicast case
            result = socket.socket( socket.AF_INET,     # Internet
                                  socket.SOCK_DGRAM,
                                  0)    # UDP
            result.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            result.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, socket.inet_aton(self.multicast_address) + socket.inet_aton(self.local_ip_address))
            try:
                result.bind( (self.local_ip_address, port) )
            except socket.error as msg:
                print("ERROR: data socket error occurred:\n%s" %msg)
                print("  Check Motive/Server mode requested mode agreement.  You requested Multicast ")
                result = None
            except socket.herror:
                print("ERROR: data socket herror occurred")
                result = None
            except socket.gaierror:
                print("ERROR: data socket gaierror occurred")
                result = None
            except socket.timeout:
                print("ERROR: data socket timeout occurred. Server not responding")
                result = None
        else:
            # Unicast case
            result = socket.socket( socket.AF_INET,     # Internet
                                  socket.SOCK_DGRAM,
                                  socket.IPPROTO_UDP)
            result.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            #result.bind( (self.local_ip_address, port) )
            try:
                result.bind( ('', 0) )
            except socket.error as msg:
                print("ERROR: data socket error occurred:\n%s" %msg)
                print("Check Motive/Server mode requested mode agreement.  You requested Unicast ")
                result = None
            except socket.herror:
                print("ERROR: data socket herror occurred")
                result = None
            except socket.gaierror:
                print("ERROR: data socket gaierror occurred")
                result = None
            except socket.timeout:
                print("ERROR: data socket timeout occurred. Server not responding")
                result = None
            
            if(self.multicast_address != "255.255.255.255"):
                result.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, socket.inet_aton(self.multicast_address) + socket.inet_aton(self.local_ip_address))

        return result

    # Unpack a rigid body object from a data packet
    def __unpack_rigid_body( self, data, major, minor, rb_num):
        offset = 0
        # print('unpacking a rigid body')
        # print(len(data))
        # print(int.from_bytes(data[offset:offset+4],byteorder='little',  signed=True))

        # ID (4 bytes)
        new_id = int.from_bytes( data[offset:offset+4], byteorder='little',  signed=True )
        offset += 4

        # Position and orientation
        pos = Vector3.unpack( data[offset:offset+12] )
        offset += 12

        rot = Quaternion.unpack( data[offset:offset+16] )
        offset += 16

        rigid_body = MoCapData.RigidBody(new_id, pos, rot)

        # print(rigid_body.id_num)
        # major = 3
        # print(major, minor)

        # RB Marker Data ( Before version 3.0.  After Version 3.0 Marker data is in description )
        if( major < 3  and major != 0) :
            # Marker count (4 bytes)
            marker_count = int.from_bytes( data[offset:offset+4], byteorder='little',  signed=True )
            offset += 4
            marker_count_range = range( 0, marker_count )

            rb_marker_list=[]
            for i in marker_count_range:
                rb_marker_list.append(MoCapData.RigidBodyMarker())

            # Marker positions
            for i in marker_count_range:
                pos = Vector3.unpack( data[offset:offset+12] )
                offset += 12
                rb_marker_list[i].pos=pos

            if major >= 2:
                # Marker ID's
                for i in marker_count_range:
                    new_id = int.from_bytes( data[offset:offset+4], byteorder='little',  signed=True )
                    offset += 4
                    rb_marker_list[i].id=new_id

                # Marker sizes
                for i in marker_count_range:
                    size = FloatValue.unpack( data[offset:offset+4] )
                    offset += 4
                    rb_marker_list[i].size=size

            for i in marker_count_range:
                rigid_body.add_rigid_body_marker(rb_marker_list[i])
        if major >= 2 :
            marker_error, = FloatValue.unpack( data[offset:offset+4] )
            offset += 4
            rigid_body.error = marker_error

        # Version 2.6 and later
        if ( ( major == 2 ) and ( minor >= 6 ) ) or major > 2 :
            param, = struct.unpack( 'h', data[offset:offset+2] )
            tracking_valid = ( param & 0x01 ) != 0
            offset += 2
            is_valid_str='False'
            if tracking_valid:
                is_valid_str = 'True'
            if tracking_valid:
                rigid_body.tracking_valid = True
            else:
                rigid_body.tracking_valid = False

        return offset, rigid_body

#Unpack Mocap Data Functions
    def __unpack_frame_prefix_data( self, data):
        offset = 0
        # Frame number (4 bytes)
        frame_number = int.from_bytes( data[offset:offset+4], byteorder='little',  signed=True )
        offset += 4
        frame_prefix_data=MoCapData.FramePrefixData(frame_number)
        return offset, frame_prefix_data

    def __unpack_data_size(self, data, major, minor):
        sizeInBytes=0
        offset=0

        if( ( (major == 4) and (minor>0) ) or (major > 4)):
            sizeInBytes = int.from_bytes( data[offset:offset+4], byteorder='little',  signed=True )
            offset += 4

        return offset, sizeInBytes

    def __unpack_legacy_other_markers( self, data, packet_size, major, minor):
        offset = 0

        # Markerset count (4 bytes)
        other_marker_count = int.from_bytes( data[offset:offset+4], byteorder='little',  signed=True )
        offset += 4

        # get data size (4 bytes)
        offset_tmp, unpackedDataSize = self.__unpack_data_size(data[offset:],major, minor)
        offset += offset_tmp

        other_marker_data = MoCapData.LegacyMarkerData()
        if(other_marker_count > 0):
            # get legacy_marker positions
            ### legacy_marker_data
            for j in range( 0, other_marker_count ):
                pos = Vector3.unpack( data[offset:offset+12] )
                offset += 12
                other_marker_data.add_pos(pos)
 
        return offset, other_marker_data

    def __unpack_marker_set_data( self, data, packet_size, major, minor):
        marker_set_data=MoCapData.MarkerSetData()
        offset = 0
        # Markerset count (4 bytes)
        marker_set_count = int.from_bytes( data[offset:offset+4], byteorder='little',  signed=True )
        offset += 4

        # get data size (4 bytes)
        offset_tmp, unpackedDataSize = self.__unpack_data_size(data[offset:],major, minor)
        offset += offset_tmp

        for i in range( 0, marker_set_count ):
            marker_data = MoCapData.MarkerData()
            # Model name
            model_name, separator, remainder = bytes(data[offset:]).partition( b'\0' )
            offset += len( model_name ) + 1
            marker_data.set_model_name(model_name)
            # Marker count (4 bytes)
            marker_count = int.from_bytes( data[offset:offset+4], byteorder='little',  signed=True )
            offset += 4
            if(marker_count < 0):
                print("WARNING: Early return.  Invalid marker count")
                offset = len(data)
                return offset, marker_set_data
            elif(marker_count > 10000):
                print("WARNING: Early return.  Marker count too high")
                offset = len(data)
                return offset, marker_set_data

            for j in range( 0, marker_count ):
                if(len(data)<(offset+12)):
                    print("WARNING: Early return.  Out of data at marker ",j," of ", marker_count)
                    offset = len(data)
                    return offset, marker_set_data
                    break
                pos = Vector3.unpack( data[offset:offset+12] )
                offset += 12
                marker_data.add_pos(pos)
            marker_set_data.add_marker_data(marker_data)

        # Unlabeled markers count (4 bytes)
        #unlabeled_markers_count = int.from_bytes( data[offset:offset+4], byteorder='little',  signed=True )
        #offset += 4
        #trace_mf( "Unlabeled Marker Count:", unlabeled_markers_count )

        #for i in range( 0, unlabeled_markers_count ):
        #    pos = Vector3.unpack( data[offset:offset+12] )
        #    offset += 12
        #    trace_mf( "\tMarker %3.1d : [%3.2f,%3.2f,%3.2f]"%( i, pos[0], pos[1], pos[2] ))
        #    marker_set_data.add_unlabeled_marker(pos)
        return offset, marker_set_data
    

    def __unpack_rigid_body_data( self, data, packet_size, major, minor):
        rigid_body_data = MoCapData.RigidBodyData()
        offset = 0
        # Rigid body count (4 bytes)
        rigid_body_count = int.from_bytes( data[offset:offset+4], byteorder='little',  signed=True )
        offset += 4

        # print('found %d rigid bodies.' % rigid_body_count)
        # print('offset: ' + str(offset))

        # get data size (4 bytes)
        offset_tmp, unpackedDataSize = self.__unpack_data_size(data[offset:],major, minor)
        offset += offset_tmp
        # print('offset: ' + str(offset))
        # print('offset_tmp: ' + str(offset_tmp))

        for i in range( 0, rigid_body_count ):
            offset_tmp, rigid_body = self.__unpack_rigid_body( data[offset:], major, minor, i )
            offset += offset_tmp
            # print('offset: ' + str(offset))
            rigid_body_data.add_rigid_body(rigid_body)

        return offset, rigid_body_data


    def __decode_marker_id(self, new_id):
        model_id = 0
        marker_id = 0
        model_id = new_id >> 16
        marker_id = new_id & 0x0000ffff
        return model_id, marker_id
    
    #DREW DID THIS
    def __unpack_suffix_data(self,data,frame_number):
        offset = 16 #I think I have 16 bytes of 0s
        #ACCORDING TO COPILOT, THE SUFFIX SHOULD BE IN THE FOLLOWING ORDER
        # Timecode (4 bytes)
        # --- Timecode (uint32) ---
        timecode = struct.unpack_from("<I", data, offset)[0]
        offset += 4

        # --- Timecode Subframe (uint32) ---
        timecode_sub = struct.unpack_from("<I", data, offset)[0]
        offset += 4

        # --- Timestamp (double, seconds) ---
        timestamp = struct.unpack_from("<d", data, offset)[0]
        offset += 8

        # --- Camera Mid Exposure Timestamp (uint64) ---
        camera_mid_ts = struct.unpack_from("<Q", data, offset)[0]
        offset += 8

        # --- Camera Data Received Timestamp (uint64) ---
        camera_recv_ts = struct.unpack_from("<Q", data, offset)[0]
        offset += 8

        # --- Transmit Timestamp (uint64) ---
        transmit_ts = struct.unpack_from("<Q", data, offset)[0]
        offset += 8

        # --- Software Latency (float) ---
        software_latency = struct.unpack_from("<f", data, offset)[0]
        offset += 4

        # print(timecode, timecode_sub)
    
        # frame_suffix_data=MoCapData.FrameSuffixData(frame_number)
        return offset, timestamp #all I care about is the timestamp for now, but I can add the rest to the FrameSuffixData class if needed
    
    # Unpack data from a motion capture frame message
    def __unpack_mocap_data( self, data : bytes, packet_size, major, minor):
        mocap_data = MoCapData.MoCapData()
        data = memoryview( data )
        offset = 0
        rel_offset = 0

        #Frame Prefix Data
        rel_offset, frame_prefix_data = self.__unpack_frame_prefix_data(data[offset:])
        offset += rel_offset
        mocap_data.set_prefix_data(frame_prefix_data)
        frame_number = frame_prefix_data.frame_number


        #Markerset Data
        rel_offset, marker_set_data =self.__unpack_marker_set_data(data[offset:], (packet_size - offset),major, minor)
        offset += rel_offset
        mocap_data.set_marker_set_data(marker_set_data)
        marker_set_count = marker_set_data.get_marker_set_count()
        unlabeled_markers_count = marker_set_data.get_unlabeled_marker_count()

        # # Legacy Other Markers
        rel_offset, legacy_other_markers =self.__unpack_legacy_other_markers(data[offset:], (packet_size - offset),major, minor)
        offset += rel_offset
        mocap_data.set_legacy_other_markers(legacy_other_markers)
        marker_set_count = legacy_other_markers.get_marker_count()
        legacy_other_markers_count = marker_set_data.get_unlabeled_marker_count()

        # Rigid Body Data
        rel_offset, rigid_body_data = self.__unpack_rigid_body_data(data[offset:], (packet_size - offset),major, minor)
        offset += rel_offset
        mocap_data.set_rigid_body_data(rigid_body_data)
        rigid_body_count = rigid_body_data.get_rigid_body_count()
        # print(rigid_body_count)
        # print(mocap_data)
        # offset = packet_size - 50 #I think the suffix is only

        #DREW DID THIS    
        rel_offset, time_stamp = self.__unpack_suffix_data(data[offset:], frame_number)
        mocap_data.timestamp = frame_number
        # print(data[offset:].hex())

        return offset, mocap_data


    def __data_thread_function(self, in_socket):
        message_id_dict={}
        data=bytearray(0)
        # 64k buffer size
        recv_buffer_size=64*1024

        # Block for input
        ret_val = [False]

        try:
            data, addr = in_socket.recvfrom( recv_buffer_size )
        except socket.error as msg:
            print("ERROR: data socket access error occurred:\n  %s" %msg)
        except socket.herror:
            print("ERROR: data socket access herror occurred")
        except socket.gaierror:
            print("ERROR: data socket access gaierror occurred")
        except socket.timeout:
            print("ERROR: data socket access timeout occurred. Server not responding")
        if len( data ) > 0 :
            #peek ahead at message_id
            message_id = get_message_id(data)
            tmp_str="mi_%1.1d"%message_id
            if tmp_str not in message_id_dict:
                message_id_dict[tmp_str]=0
            message_id_dict[tmp_str] += 1
            
            message_id, mocap_data = self.__process_message(data)

            ret_val = []
            ret_val = [mocap_data.timestamp] #drew did this
            for item in mocap_data.rigid_body_data.rigid_body_list:
                id = item.id_num
                pos = item.pos
                rot = item.rot
                mini_ret_val = [True, id, pos, rot]
                ret_val.append(mini_ret_val)
            # id = mocap_data.rigid_body_data.rigid_body_list[0].id_num
            # pos = mocap_data.rigid_body_data.rigid_body_list[0].pos
            # rot = mocap_data.rigid_body_data.rigid_body_list[0].rot

            # ret_val = [True, id, pos, rot]


        return ret_val

    def __process_message( self, data : bytes):
        #return message ID
        if not self.preset_major:
            major = self.get_major()
            minor = self.get_minor()
        else:
            major = self.preset_major
            minor = self.preset_minor

        # print('processing a new message')

        message_id = get_message_id(data)

        packet_size = int.from_bytes( data[2:4], byteorder='little',  signed=True )

        # print('packet size', packet_size)

        #skip the 4 bytes for message ID and packet_size
        offset = 4
        if message_id == self.NAT_FRAMEOFDATA :

            offset_tmp, mocap_data = self.__unpack_mocap_data( data[offset:], packet_size, major, minor )
            offset += offset_tmp
            # print("MoCap Frame: %d\n"%(mocap_data.prefix_data.frame_number))
            # get a string version of the data for output
            mocap_data_str=mocap_data.get_as_string()
            # print(mocap_data_str)


        else:
            print('There was a different kind of message and I just ignored it.')

        return message_id, mocap_data

    def send_request( self, in_socket, command, command_str, address ):
        # Compose the message in our known message format
        packet_size = 0
        if command == self.NAT_REQUEST_MODELDEF or command == self.NAT_REQUEST_FRAMEOFDATA :
            packet_size = 0
            command_str = ""
        elif command == self.NAT_REQUEST :
            packet_size = len( command_str ) + 1
        elif command == self.NAT_CONNECT :
            tmp_version=[4,1,0,0]
            print("NAT_CONNECT to Motive with %d %d %d %d\n"%(
                tmp_version[0],
                tmp_version[1],
                tmp_version[2],
                tmp_version[3]
            ))
            #allocate a byte array for 270 bytes
            # to connect with a specific version
            # The first 4 bytes spell out "Ping"
            command_str = []
            command_str = [0 for i in range(270)]
            command_str[0] =80
            command_str[1] =105
            command_str[2] =110
            command_str[3] =103
            command_str[264] =0
            command_str[265] =tmp_version[0]
            command_str[266] =tmp_version[1]
            command_str[267] =tmp_version[2]
            command_str[268] =tmp_version[3]
            packet_size = len( command_str ) + 1
        elif command == self.NAT_KEEPALIVE:
            packet_size = 0
            command_str = ""

        data = command.to_bytes( 2, byteorder='little',  signed=True )
        data += packet_size.to_bytes( 2, byteorder='little',  signed=True )

        if command == self.NAT_CONNECT :
            data+=bytearray(command_str)
        else:
            data += command_str.encode( 'utf-8' )
        data += b'\0'

        return in_socket.sendto( data, address )

    def send_command( self, command_str):
        #print("Send command %s"%command_str)
        nTries = 3
        ret_val = -1
        while nTries:
            nTries -= 1
            ret_val = self.send_request( self.command_socket, self.NAT_REQUEST, command_str,  (self.server_ip_address, self.command_port) )
            if (ret_val != -1):
                break;
        return ret_val

        #return self.send_request(self.data_socket,    self.NAT_REQUEST, command_str,  (self.server_ip_address, self.command_port) )

    def send_commands(self,tmpCommands, print_results: bool =True):
        for sz_command in tmpCommands:
            return_code = self.send_command(sz_command)
            if(print_results):
                print("Command: %s - return_code: %d"% (sz_command, return_code) )

    def send_keep_alive(self,in_socket, server_ip_address, server_port):
        return self.send_request(in_socket, self.NAT_KEEPALIVE, "", (server_ip_address, server_port))

    def get_command_port(self):
        return self.command_port

    def refresh_configuration(self):
        #query for application configuration
        #print("Request current configuration")
        sz_command = "Bitstream"
        return_code = self.send_command(sz_command)
        time.sleep(0.5)

    def get_application_name(self):
        return self.__application_name

    def get_nat_net_requested_version(self):
        return self.__nat_net_requested_version

    def get_nat_net_version_server(self):
        return self.__nat_net_stream_version_server

    def get_server_version(self):
        return self.__server_version

    def initialize_optitrack_comms(self):
        print('intitializing optitrack comms')
        #Create the data socket
        self.data_socket = self.__create_data_socket( self.data_port )
        if self.data_socket is None :
            print( "Could not open data channel" )
            return False
        return True
        
    def get_pos(self):

        pos = self.__data_thread_function(self.data_socket)
        #[T/F for success, id, pos, rot]

        return pos


    def shutdown(self):
        print("shutdown called")
        self.stop_threads = True
        # closing sockets causes blocking recvfrom to throw
        # an exception and break the loop
        self.data_socket.close()


