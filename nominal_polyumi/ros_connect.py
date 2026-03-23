import asyncio
import json
import logging
import struct
from datetime import datetime, timezone
from urllib.parse import urlparse, urlunparse

import connect_python
import numpy as np
import websockets
from mcap_ros2.decoder import DecoderFactory

logging.basicConfig(level=logging.INFO)

TIMESERIES_MSGS = [
    'std_msgs/msg/Bool',
    'std_msgs/msg/Byte',
    'std_msgs/msg/Float32',
    'std_msgs/msg/Float64',
    'std_msgs/msg/Int16',
    'std_msgs/msg/Int32',
    'std_msgs/msg/Int64',
    'std_msgs/msg/Int8',
    'std_msgs/msg/UInt16',
    'std_msgs/msg/UInt32',
    'std_msgs/msg/UInt64',
    'std_msgs/msg/UInt8',
]

IMG_MSGS = [
    'sensor_msgs/msg/Image',
]


class Schema:
    id = 0
    name = ''
    encoding = ''
    data = bytearray


prev_vid = ''


def normalize_websocket_url(raw_url: str, logger) -> str:
    parsed = urlparse(raw_url)
    if parsed.hostname != '0.0.0.0':
        return raw_url

    port = f':{parsed.port}' if parsed.port else ''
    auth = ''
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth += f':{parsed.password}'
        auth += '@'

    normalized = urlunparse(
        (
            parsed.scheme,
            f'{auth}localhost{port}',
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )
    logger.warn(f'Rewriting websocket URL from {raw_url} to {normalized}')
    return normalized


def process_img(client: connect_python.Client, msg, time_now, logger):
    global prev_vid

    client.clear_frame_buffer('frame_buffer')

    prev_vid = msg.encoding
    match msg.encoding:
        case 'rgb8':
            raw_data = np.frombuffer(msg.data, dtype=np.uint8)
            frame = raw_data.reshape((msg.height, msg.width, 3))
            rgb_array = frame.reshape(-1).tobytes()
        case 'mono8':
            raw_data = np.frombuffer(msg.data, dtype=np.uint8)
            frame = raw_data.reshape((msg.height, msg.width))
            rgb_array = np.repeat(frame.reshape(-1), 3).tobytes()
        case '16UC1':
            raw_data = np.frombuffer(msg.data, dtype=np.uint16)
            frame = raw_data.reshape((msg.height, msg.width))
            frame = (frame / 256).astype(np.uint8)
            rgb_array = np.repeat(frame.reshape(-1), 3).tobytes()
        case _:
            logger.warn(f'Do not support {msg.encoding}')
            return

    client.stream_rgb('frame_buffer', time_now, msg.width, rgb_array)


async def run_client(client: connect_python.Client):
    logger = connect_python.get_logger(__name__)

    all_values = client.get_values()

    all_std_topics = set()
    all_img_topics = set()
    all_topics = set()

    dropdown_subs = set()
    for key, value in all_values.items():
        if '_sub' in key:
            dropdown_subs.add(key)
    subscribed_topics = []

    client.clear_all_streams()
    client.clear_frame_buffer('frame_buffer')

    websocket_url = normalize_websocket_url(
        client.get_value(id='websocket_url'), logger
    )

    channels = {}
    import sys

    logger.info(f'Websockets version {websockets.__version__}, {websockets.__file__}')
    logger.info(f'Python executable: {sys.executable}')
    async with websockets.connect(
        websocket_url,
        subprotocols=['foxglove.websocket.v1', 'foxglove.sdk.v1'],
        extra_headers={'Origin': 'https://app.foxglove.dev'},
        compression=None,
        max_size=None,
    ) as ws:
        while True:
            subscribed_topics = [client.get_value(id=topic) for topic in dropdown_subs]

            try:
                message = await ws.recv()

                if isinstance(message, str):
                    # WebSocket Text
                    data = json.loads(message)
                    if data.get('op') == 'serverInfo':
                        logger.info(f'Server: {data.get("name", "unknown")}')
                        logger.info(f'Capabilities: {data.get("capabilities", [])}')

                    elif data.get('op') == 'status':
                        logger.info(f'status: {json.dumps(data, indent=2)}')
                    elif data.get('op') == 'advertise':
                        logger.info(f'advertise: {json.dumps(data, indent=2)}')

                        for i, channel in enumerate(data['channels']):
                            channels[i] = channel
                            await ws.send(
                                json.dumps(
                                    {
                                        'op': 'subscribe',
                                        'subscriptions': [
                                            {
                                                'id': i,
                                                'channelId': channel['id'],
                                                'topic': channel['topic'],
                                            }
                                        ],
                                    }
                                )
                            )

                            if channel['schemaName'] in TIMESERIES_MSGS:
                                all_std_topics.add(channel['topic'])
                            elif channel['schemaName'] in IMG_MSGS:
                                all_img_topics.add(channel['topic'])

                            all_topics.add(
                                f'{channel["topic"]}: \n\t{channel["schemaName"]}\n'
                            )

                    elif data.get('op') == 'advertiseServices':
                        logger.info(f'advertiseServices: {json.dumps(data, indent=2)}')
                    elif data.get('op') == 'messageData':
                        logger.info(f'messageData: {json.dumps(data, indent=2)}')
                    else:
                        logger.info(
                            f'unexpected response: {json.dumps(data, indent=2)}'
                        )
                elif isinstance(message, bytes):
                    # WebSocket Binary

                    # Read opcode (1 byte)
                    offset = 0
                    _opcode = struct.unpack('<B', message[offset : offset + 1])[0]
                    offset += 1
                    subscription_id = struct.unpack('<I', message[offset : offset + 4])[
                        0
                    ]
                    offset += 4
                    _timestamp_ns = struct.unpack('<Q', message[offset : offset + 8])[0]
                    offset += 8
                    payload = message[offset:]

                    decoder_factory = DecoderFactory()
                    schema = Schema()

                    topic = channels[subscription_id]['topic']

                    if topic not in subscribed_topics:
                        continue

                    schema.encoding = channels[subscription_id]['schemaEncoding']
                    schema.name = channels[subscription_id]['schemaName']
                    schema.id = channels[subscription_id]['id']
                    schema.data = channels[subscription_id]['schema'].encode('utf-8')

                    try:
                        decoder = decoder_factory.decoder_for(
                            channels[subscription_id]['encoding'], schema
                        )
                        msg = decoder(payload)
                    except Exception:
                        continue

                    time_now = datetime.now(tz=timezone.utc).timestamp()

                    if client.get_value(id='std_0_sub') == topic:
                        client.stream('std_0_stream', time_now, msg.data)
                    elif client.get_value(id='std_1_sub') == topic:
                        client.stream('std_1_stream', time_now, msg.data)
                    elif client.get_value(id='img_sub') == topic:
                        process_img(client, msg, time_now, logger)

                client.set_value('topic_list', list(all_topics))
                client.set_dropdown_options(id='std_0_sub', options=all_std_topics)
                client.set_dropdown_options(id='std_1_sub', options=all_std_topics)
                client.set_dropdown_options(id='img_sub', options=all_img_topics)
            except Exception as e:
                logger.info(f'Error {e}')


@connect_python.main
def run(client: connect_python.Client):
    asyncio.run(run_client(client))


if __name__ == '__main__':
    run()
