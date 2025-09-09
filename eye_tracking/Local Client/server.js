const http = require('http');
const WebSocket = require('ws');
const fs = require('fs');
const path = require('path');

// Generate a unique file name with the current date and time
function generateFileName(baseName) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-'); // Format: YYYY-MM-DDTHH-MM-SS
    const ext = path.extname(baseName); // Get the file extension
    const name = path.basename(baseName, ext); // Get the base name without extension
    return `${name}_${timestamp}${ext}`; // Combine name, timestamp, and extension
}

// Filepath for the eye tracking data
let filePath = path.join(__dirname, 'eye_data.csv');

// If the file exists, create a new file with a timestamped name
if (fs.existsSync(filePath)) {
    const newFilePath = generateFileName(filePath);
    console.log(`The file "${filePath}" already exists. Data will be saved to "${newFilePath}".`);
    filePath = newFilePath;
} else {
    console.log(`The file "${filePath}" does not exist. Creating a new file.`);
}

// Create the file with headers
fs.writeFileSync(filePath, 'timestamp,datetimestamp,x_axis,y_axis\n', 'utf8');

// Initialize the HTTP server
const server = http.createServer((req, res) => {pp
    if (req.url === '/') {
        fs.readFile(path.join(__dirname, 'index.html'), (err, data) => {
            if (err) {
                res.writeHead(500);
                res.end('Error loading index.html');
                return;
            }
            res.writeHead(200, { 'Content-Type': 'text/html' });
            res.end(data);
        });
    } else if (req.url.endsWith('.js')) {
        fs.readFile(path.join(__dirname, req.url), (err, data) => {
            if (err) {
                res.writeHead(404);
                res.end('File not found');
                return;
            }
            res.writeHead(200, { 'Content-Type': 'application/javascript' });
            res.end(data);
        });
    } else {
        res.writeHead(404);
        res.end('Not Found');
    }
});

// Initialize the WebSocket server
const wss = new WebSocket.Server({ server });

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

// Start the server
const PORT = 8080;
server.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});
