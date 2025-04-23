const { app, BrowserWindow, Tray, Menu, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const axios = require('axios');
const log = require('electron-log');
const Store = require('electron-store');

// Configure logging
log.transports.file.level = 'info';
log.info('Application starting...');

// Create a store for application settings
const store = new Store({
  name: 'ai-assistant-config',
  defaults: {
    startOnBoot: true,
    runInBackground: true,
    firstRun: true,
    serverPort: 8000,
    clientPort: 3000
  }
});

let mainWindow;
let tray;
let backendProcess;
let isQuitting = false;
const serverPort = store.get('serverPort');
const clientPort = store.get('clientPort');
const serverUrl = `http://localhost:${serverPort}`;
const clientUrl = `http://localhost:${clientPort}`;
let serverReady = false;

// Handle creating/removing shortcuts on Windows when installing/uninstalling
if (require('electron-squirrel-startup')) {
  app.quit();
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    icon: path.join(__dirname, 'icons/icon.png'),
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      preload: path.join(__dirname, 'preload.js')
    }
  });

  // In production, load the bundled React app
  // In development, load from the React dev server
  if (app.isPackaged) {
    mainWindow.loadFile(path.join(__dirname, '../build/index.html'));
  } else {
    mainWindow.loadURL(clientUrl);
    mainWindow.webContents.openDevTools(); // Open DevTools in development
  }

  // Handle window close event
  mainWindow.on('close', (event) => {
    if (!isQuitting && store.get('runInBackground')) {
      event.preventDefault();
      mainWindow.hide();
      return false;
    }
    
    // Always save window state before closing
    store.set('windowBounds', mainWindow.getBounds());
  });

  // Send the backend URL to the renderer process
  mainWindow.webContents.on('did-finish-load', () => {
    mainWindow.webContents.send('backend-url', serverUrl);
  });
}

function startBackendServer() {
  return new Promise((resolve, reject) => {
    log.info('Starting backend server...');
    
    // Path to the Python executable and backend script
    // In production, the backend is bundled as part of the app resources
    const isProd = app.isPackaged;
    const pythonPath = isProd ? path.join(process.resourcesPath, 'backend/venv/Scripts/python.exe') : 'python';
    const scriptPath = isProd ? path.join(process.resourcesPath, 'backend/main.py') : path.join(__dirname, '../backend/main.py');
    
    log.info(`Python path: ${pythonPath}`);
    log.info(`Script path: ${scriptPath}`);
    
    // Launch the backend server
    backendProcess = spawn(pythonPath, [scriptPath]);
    
    backendProcess.stdout.on('data', (data) => {
      const output = data.toString().trim();
      log.info(`Backend: ${output}`);
      
      // Check if server is ready
      if (output.includes('Application startup complete') || output.includes('Uvicorn running on')) {
        serverReady = true;
        resolve();
      }
    });
    
    backendProcess.stderr.on('data', (data) => {
      log.error(`Backend error: ${data.toString().trim()}`);
    });
    
    backendProcess.on('close', (code) => {
      log.info(`Backend server exited with code ${code}`);
      backendProcess = null;
      if (code !== 0 && !isQuitting) {
        // Server crashed, show error dialog
        if (mainWindow) {
          dialog.showErrorBox(
            'Server Error',
            'The backend server has crashed. Please restart the application.'
          );
        }
      }
    });
    
    // Check if server starts successfully
    setTimeout(() => {
      if (!serverReady) {
        reject(new Error('Backend server failed to start'));
      }
    }, 10000); // 10 second timeout
  });
}

function createTray() {
  tray = new Tray(path.join(__dirname, 'icons/icon.png'));
  const contextMenu = Menu.buildFromTemplate([
    { label: 'Open AI Assistant', click: () => { mainWindow.show(); } },
    { type: 'separator' },
    { label: 'Start on Boot', type: 'checkbox', checked: store.get('startOnBoot'), click: () => {
      const newValue = !store.get('startOnBoot');
      store.set('startOnBoot', newValue);
      setupAutoLaunch(newValue);
    }},
    { label: 'Run in Background', type: 'checkbox', checked: store.get('runInBackground'), click: () => {
      store.set('runInBackground', !store.get('runInBackground'));
    }},
    { type: 'separator' },
    { label: 'Quit', click: () => { 
      isQuitting = true;
      app.quit(); 
    }}
  ]);
  
  tray.setToolTip('Personal AI Assistant');
  tray.setContextMenu(contextMenu);
  
  tray.on('click', () => {
    mainWindow.isVisible() ? mainWindow.hide() : mainWindow.show();
  });
}

function setupAutoLaunch(enable) {
  const AutoLaunch = require('auto-launch');
  const appAutoLauncher = new AutoLaunch({
    name: 'Personal AI Assistant',
    path: app.getPath('exe')
  });
  
  if (enable) {
    appAutoLauncher.enable();
  } else {
    appAutoLauncher.disable();
  }
}

function checkServerHealth() {
  axios.get(`${serverUrl}/health`)
    .then(response => {
      log.info('Server health check successful');
      mainWindow.webContents.send('server-status', { status: 'connected' });
    })
    .catch(error => {
      log.error('Server health check failed', error.message);
      mainWindow.webContents.send('server-status', { status: 'disconnected' });
    });
}

async function initApp() {
  try {
    // Start the backend server
    await startBackendServer();
    
    // Create the browser window
    createWindow();
    
    // Create the system tray icon
    createTray();
    
    // Set up auto-launch if enabled
    if (store.get('startOnBoot')) {
      setupAutoLaunch(true);
    }
    
    // Set up periodic health checks
    setInterval(checkServerHealth, 10000);
    
    // Show first-run dialog if needed
    if (store.get('firstRun')) {
      showFirstRunDialog();
      store.set('firstRun', false);
    }
    
  } catch (error) {
    log.error('Initialization error:', error);
    dialog.showErrorBox(
      'Initialization Error',
      `Failed to start the application: ${error.message}`
    );
    isQuitting = true;
    app.quit();
  }
}

function showFirstRunDialog() {
  dialog.showMessageBox(mainWindow, {
    type: 'info',
    title: 'Welcome to Personal AI Assistant',
    message: 'Thank you for installing Personal AI Assistant!',
    detail: 'This assistant will learn from your activities to automate tasks. You can control what is monitored in the Settings page. The application will run in the background and can be accessed from the system tray.',
    buttons: ['Get Started'],
    icon: path.join(__dirname, 'icons/icon.png')
  });
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
app.on('ready', initApp);

// Quit when all windows are closed, except on macOS.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  // On macOS it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  } else {
    mainWindow.show();
  }
});

// Clean up before quitting
app.on('before-quit', () => {
  isQuitting = true;
  
  // Kill the backend server process
  if (backendProcess) {
    try {
      if (process.platform === 'win32') {
        // On Windows, use taskkill to make sure all child processes are terminated
        spawn('taskkill', ['/pid', backendProcess.pid, '/t', '/f']);
      } else {
        // On Linux/Mac, use process group to kill children
        process.kill(-backendProcess.pid);
      }
    } catch (error) {
      log.error('Error killing backend process:', error);
    }
  }
});

// IPC handlers
ipcMain.on('toggle-monitoring', (event, enabled) => {
  log.info(`Toggle monitoring: ${enabled}`);
  // Send API call to backend
  axios.post(`${serverUrl}/settings/monitoring`, { enabled })
    .then(() => {
      event.reply('monitoring-status', { enabled });
    })
    .catch(error => {
      log.error('Error toggling monitoring:', error);
      event.reply('monitoring-status', { error: error.message });
    });
});

ipcMain.on('show-app', () => {
  if (mainWindow) {
    mainWindow.show();
  }
});
