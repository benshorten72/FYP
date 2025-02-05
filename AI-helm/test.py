import paho.mqtt.subscribe as subscribe
msg = subscribe.simple("result", hostname="edgex-mqtt-broker", port=1883)
print("%s %s" % (msg.topic, msg.payload))
