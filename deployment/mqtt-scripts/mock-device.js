// docker run --rm --name=mqtt-scripts \
// -v /Users/benshorten/Desktop/fyp/helm-edge/edgex-helm/deployment/mqtt-scripts:/scripts --network testbed \
// dersimn/mqtt-scripts --dir /scripts

function getRandomFloat(min, max) {
    return Math.random() * (max - min) + min;
}

const deviceName = "my-custom-device";
let message = "test-message";
let json = {"name" : "My JSON"};

// DataSender sends async value to MQTT broker every 15 seconds
schedule('*/15 * * * * *', ()=>{
    var data = {};
    data.randnum = getRandomFloat(25,29).toFixed(1);
    data.ping = "pong"
    data.message = "Hello World"

    publish( 'incoming/data/my-custom-device/values', JSON.stringify(data));
});

// CommandHandler receives commands and sends response to MQTT broker
// 1. Receive the reading request, then return the response
// 2. Receive the set request, then change the device value
subscribe( "command/my-custom-device/#" , (topic, val) => {
    const words = topic.split('/');
    var cmd = words[2];
    var method = words[3];
    var uuid = words[4];
    var response = {};
    var data = val;

    if (method == "set") {
        switch(cmd) {
            case "message":
                message = data[cmd];
                break;
            case "json":
                json = data[cmd];
                break;
        }
    }else{
        switch(cmd) {
            case "ping":
                response.ping = "pong";
                break;
            case "message":
                response.message = message;
                break;
            case "randnum":
                response.randnum = 12.123;
                break;
            case "json":
                response.json = json;
                break;
        }
    }
    var sendTopic ="command/response/"+ uuid;
    publish( sendTopic, JSON.stringify(response));
});
