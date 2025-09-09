const WebSocket = require('ws');
const fs = require('fs');
const path = require('path');
const readline = require('readline');

// Filepath for eye_data.csv
const filePath = path.join(__dirname, 'eye_data.csv');

// Function to prompt the user
function promptUser(question) {
    return new Promise((resolve) => {
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout,
        });
        rl.question(question, (answer) => {
            rl.close();
            resolve(answer.trim().toLowerCase());
        });
    });
}

// Initialize WebSocket server only
const wss = new WebSocket.Server({ port: 8080 });

wss.on('connection', (ws) => {
    console.log('Client connected.');

    ws.on('message', (message) => {
        try {
            const data = JSON.parse(message);
            const csvLine = `${data.timestamp},${data.datetimestamp},${data.x_axis},${data.y_axis}\n`;
            fs.appendFileSync(filePath, csvLine, 'utf8');
        } catch (error) {
            console.error('Error processing message:', error);
        }
    });

    ws.on('close', () => {
        console.log('Client disconnected.');
    });
});

// Main logic
(async () => {
    // Check if the file exists and prompt the user
    if (fs.existsSync(filePath)) {
        const answer = await promptUser(
            `The file "eye_data.csv" already exists. Do you want to clear the previous data? (yes/no): `
        );
        if (answer === 'yes') {
            // Clear the file and add headers
            fs.writeFileSync(filePath, 'timestamp,datetimestamp,x_axis,y_axis\n', 'utf8');
            console.log('Previous data cleared.');
        } else {
            console.log('Keeping the previous data.');
        }
    } else {
        // Create the file with headers if it doesn't exist
        fs.writeFileSync(filePath, 'timestamp,datetimestamp,x_axis,y_axis\n', 'utf8');
        console.log('File created with headers.');
    }

    console.log('WebSocket server running on ws://localhost:8080');
})();
