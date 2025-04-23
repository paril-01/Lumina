import React, { useState, useEffect, useContext } from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  CircularProgress,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Divider,
  Switch,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  TextField,
  Card,
  CardContent,
  CardActions,
  Chip,
  Tooltip
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import EditIcon from '@mui/icons-material/Edit';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import TuneIcon from '@mui/icons-material/Tune';
import AppsIcon from '@mui/icons-material/Apps';
import AlertContext from '../context/AlertContext';

function TaskManager() {
  const { showAlert } = useContext(AlertContext);
  const [loading, setLoading] = useState(true);
  const [tasks, setTasks] = useState([]);
  const [openDialog, setOpenDialog] = useState(false);
  const [confirmDialog, setConfirmDialog] = useState({
    open: false,
    title: '',
    message: '',
    onConfirm: null
  });
  const [selectedTask, setSelectedTask] = useState(null);
  const [newTaskData, setNewTaskData] = useState({
    task_name: '',
    task_type: 'keyboard',
    is_active: true
  });

  useEffect(() => {
    // Simulate loading tasks from API
    const timer = setTimeout(() => {
      const mockTasks = generateMockTasks();
      setTasks(mockTasks);
      setLoading(false);
    }, 1500);
    
    return () => clearTimeout(timer);
  }, []);

  // Generate mock task data
  const generateMockTasks = () => {
    return [
      {
        id: 1,
        task_name: 'Open email after calendar check',
        task_type: 'application',
        trigger_conditions: {
          app_closed: 'Calendar',
          time_range: 'Morning (8:00-10:00)',
          days: 'Weekdays'
        },
        actions: [
          { type: 'application', subtype: 'launch', application: 'Outlook' }
        ],
        is_active: true,
        confidence_level: 87,
        execution_count: 18,
        last_executed: '2025-04-22T08:45:22'
      },
      {
        id: 2,
        task_name: 'Format spreadsheet on open',
        task_type: 'keyboard',
        trigger_conditions: {
          app_launched: 'Excel',
          file_type: '.xlsx'
        },
        actions: [
          { type: 'keyboard', subtype: 'hotkey', keys: ['ctrl', 'a'] },
          { type: 'keyboard', subtype: 'hotkey', keys: ['alt', 'h', 'f', 'c'] }
        ],
        is_active: true,
        confidence_level: 76,
        execution_count: 12,
        last_executed: '2025-04-21T14:12:05'
      },
      {
        id: 3,
        task_name: 'Start music while coding',
        task_type: 'system',
        trigger_conditions: {
          app_launched: 'Visual Studio Code',
          idle_time: '2 minutes'
        },
        actions: [
          { type: 'application', subtype: 'launch', application: 'Spotify' },
          { type: 'system', subtype: 'notification', message: 'Started your coding playlist' }
        ],
        is_active: false,
        confidence_level: 65,
        execution_count: 5,
        last_executed: '2025-04-18T16:22:33'
      }
    ];
  };

  const handleToggleTask = (taskId) => {
    setTasks(tasks.map(task => 
      task.id === taskId ? { ...task, is_active: !task.is_active } : task
    ));
    
    const task = tasks.find(t => t.id === taskId);
    if (task) {
      showAlert(`Task "${task.task_name}" ${!task.is_active ? 'activated' : 'deactivated'}`, 'success');
    }
  };

  const handleDeleteTask = (taskId) => {
    const task = tasks.find(t => t.id === taskId);
    if (!task) return;
    
    setConfirmDialog({
      open: true,
      title: 'Delete Task',
      message: `Are you sure you want to delete "${task.task_name}"?`,
      onConfirm: () => {
        setTasks(tasks.filter(task => task.id !== taskId));
        showAlert('Task deleted successfully', 'success');
        setConfirmDialog({ ...confirmDialog, open: false });
      }
    });
  };

  const handleEditTask = (task) => {
    setSelectedTask(task);
    setNewTaskData({
      task_name: task.task_name,
      task_type: task.task_type,
      is_active: task.is_active
    });
    setOpenDialog(true);
  };

  const handleRunTask = (taskId) => {
    const task = tasks.find(t => t.id === taskId);
    if (!task) return;
    
    showAlert(`Executing task: ${task.task_name}`, 'info');
    
    // Simulate task execution
    setTimeout(() => {
      setTasks(tasks.map(t => 
        t.id === taskId ? {
          ...t,
          execution_count: t.execution_count + 1,
          last_executed: new Date().toISOString()
        } : t
      ));
      showAlert(`Task "${task.task_name}" executed successfully`, 'success');
    }, 1500);
  };

  const handleOpenNewTaskDialog = () => {
    setSelectedTask(null);
    setNewTaskData({
      task_name: '',
      task_type: 'keyboard',
      is_active: true
    });
    setOpenDialog(true);
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
  };

  const handleInputChange = (e) => {
    const { name, value, checked } = e.target;
    setNewTaskData({
      ...newTaskData,
      [name]: name === 'is_active' ? checked : value
    });
  };

  const handleSaveTask = () => {
    if (!newTaskData.task_name.trim()) {
      showAlert('Task name is required', 'error');
      return;
    }
    
    if (selectedTask) {
      // Update existing task
      setTasks(tasks.map(task => 
        task.id === selectedTask.id ? {
          ...task,
          task_name: newTaskData.task_name,
          task_type: newTaskData.task_type,
          is_active: newTaskData.is_active
        } : task
      ));
      showAlert('Task updated successfully', 'success');
    } else {
      // Create new task
      const newTask = {
        id: Math.max(0, ...tasks.map(t => t.id)) + 1,
        task_name: newTaskData.task_name,
        task_type: newTaskData.task_type,
        trigger_conditions: {
          // Default trigger conditions
          app_launched: 'Any',
          time_range: 'Any time',
          days: 'All days'
        },
        actions: [
          // Default action
          { type: newTaskData.task_type, subtype: 'notification', message: 'Task executed' }
        ],
        is_active: newTaskData.is_active,
        confidence_level: 50, // Default confidence
        execution_count: 0,
        last_executed: null
      };
      
      setTasks([...tasks, newTask]);
      showAlert('New task created successfully', 'success');
    }
    
    setOpenDialog(false);
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'Never';
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const getTaskTypeIcon = (type) => {
    switch (type) {
      case 'keyboard':
        return <TuneIcon />;
      case 'application':
        return <AppsIcon />;
      case 'system':
        return <AutoFixHighIcon />;
      case 'temporal':
        return <AccessTimeIcon />;
      default:
        return <AutoFixHighIcon />;
    }
  };

  const getConfidenceLevelColor = (level) => {
    if (level >= 80) return 'success';
    if (level >= 60) return 'warning';
    return 'error';
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1">
          Task Automation
        </Typography>
        <Button
          variant="contained"
          color="primary"
          startIcon={<AddIcon />}
          onClick={handleOpenNewTaskDialog}
        >
          New Task
        </Button>
      </Box>
      
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
          <CircularProgress />
        </Box>
      ) : (
        <>
          <Grid container spacing={3}>
            {/* Active Tasks */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Active Tasks
                </Typography>
                <Divider sx={{ mb: 2 }} />
                
                {tasks.some(task => task.is_active) ? (
                  <List>
                    {tasks.filter(task => task.is_active).map((task) => (
                      <Card key={task.id} sx={{ mb: 2 }}>
                        <CardContent>
                          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                            <Box sx={{ mr: 1 }}>
                              {getTaskTypeIcon(task.task_type)}
                            </Box>
                            <Typography variant="h6" sx={{ flex: 1 }}>
                              {task.task_name}
                            </Typography>
                            <Chip 
                              label={`${task.confidence_level}%`} 
                              size="small" 
                              color={getConfidenceLevelColor(task.confidence_level)} 
                            />
                          </Box>
                          
                          <Typography variant="body2" color="textSecondary" gutterBottom>
                            Trigger: {Object.entries(task.trigger_conditions)
                              .map(([key, value]) => `${key.replace(/_/g, ' ')}: ${value}`)
                              .join(', ')}
                          </Typography>
                          
                          <Typography variant="body2" color="textSecondary">
                            Last executed: {formatDate(task.last_executed)}
                          </Typography>
                        </CardContent>
                        <CardActions sx={{ justifyContent: 'flex-end' }}>
                          <Tooltip title="Run Now">
                            <IconButton size="small" onClick={() => handleRunTask(task.id)} color="primary">
                              <PlayArrowIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Edit">
                            <IconButton size="small" onClick={() => handleEditTask(task)} color="primary">
                              <EditIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Deactivate">
                            <Switch
                              checked={task.is_active}
                              onChange={() => handleToggleTask(task.id)}
                              size="small"
                            />
                          </Tooltip>
                        </CardActions>
                      </Card>
                    ))}
                  </List>
                ) : (
                  <Box sx={{ textAlign: 'center', py: 3 }}>
                    <Typography variant="body1" color="textSecondary">
                      No active tasks
                    </Typography>
                    <Button 
                      variant="outlined" 
                      startIcon={<AddIcon />} 
                      sx={{ mt: 2 }}
                      onClick={handleOpenNewTaskDialog}
                    >
                      Create New Task
                    </Button>
                  </Box>
                )}
              </Paper>
            </Grid>
            
            {/* Inactive Tasks */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Inactive Tasks
                </Typography>
                <Divider sx={{ mb: 2 }} />
                
                {tasks.some(task => !task.is_active) ? (
                  <List>
                    {tasks.filter(task => !task.is_active).map((task) => (
                      <Card key={task.id} sx={{ mb: 2, opacity: 0.7 }}>
                        <CardContent>
                          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                            <Box sx={{ mr: 1 }}>
                              {getTaskTypeIcon(task.task_type)}
                            </Box>
                            <Typography variant="h6" sx={{ flex: 1 }}>
                              {task.task_name}
                            </Typography>
                            <Chip 
                              label={`${task.confidence_level}%`} 
                              size="small" 
                              color={getConfidenceLevelColor(task.confidence_level)} 
                            />
                          </Box>
                          
                          <Typography variant="body2" color="textSecondary" gutterBottom>
                            Trigger: {Object.entries(task.trigger_conditions)
                              .map(([key, value]) => `${key.replace(/_/g, ' ')}: ${value}`)
                              .join(', ')}
                          </Typography>
                          
                          <Typography variant="body2" color="textSecondary">
                            Last executed: {formatDate(task.last_executed)}
                          </Typography>
                        </CardContent>
                        <CardActions sx={{ justifyContent: 'flex-end' }}>
                          <Tooltip title="Edit">
                            <IconButton size="small" onClick={() => handleEditTask(task)} color="primary">
                              <EditIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Delete">
                            <IconButton size="small" onClick={() => handleDeleteTask(task.id)} color="error">
                              <DeleteIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Activate">
                            <Switch
                              checked={task.is_active}
                              onChange={() => handleToggleTask(task.id)}
                              size="small"
                            />
                          </Tooltip>
                        </CardActions>
                      </Card>
                    ))}
                  </List>
                ) : (
                  <Box sx={{ textAlign: 'center', py: 3 }}>
                    <Typography variant="body1" color="textSecondary">
                      No inactive tasks
                    </Typography>
                  </Box>
                )}
              </Paper>
            </Grid>
          </Grid>
        </>
      )}
      
      {/* New/Edit Task Dialog */}
      <Dialog open={openDialog} onClose={handleCloseDialog} maxWidth="sm" fullWidth>
        <DialogTitle>{selectedTask ? 'Edit Task' : 'Create New Task'}</DialogTitle>
        <DialogContent>
          <DialogContentText>
            {selectedTask ? 
              'Edit the task details below.' :
              'Configure your new automated task by entering the details below.'
            }
          </DialogContentText>
          
          <TextField
            margin="dense"
            label="Task Name"
            name="task_name"
            value={newTaskData.task_name}
            onChange={handleInputChange}
            fullWidth
            variant="outlined"
            required
            sx={{ mt: 2 }}
          />
          
          <TextField
            select
            margin="dense"
            label="Task Type"
            name="task_type"
            value={newTaskData.task_type}
            onChange={handleInputChange}
            fullWidth
            variant="outlined"
            sx={{ mt: 2 }}
            SelectProps={{
              native: true
            }}
          >
            <option value="keyboard">Keyboard Automation</option>
            <option value="application">Application Control</option>
            <option value="system">System Operation</option>
            <option value="temporal">Time-based Task</option>
          </TextField>
          
          <Box sx={{ display: 'flex', alignItems: 'center', mt: 2 }}>
            <Typography variant="body1" sx={{ mr: 2 }}>
              Active:
            </Typography>
            <Switch
              name="is_active"
              checked={newTaskData.is_active}
              onChange={handleInputChange}
            />
          </Box>
          
          <Typography variant="body2" color="textSecondary" sx={{ mt: 2 }}>
            {selectedTask ? 
              'Additional task configuration options will be maintained.' :
              'After creating the task, you will be able to configure triggers and actions in detail.'
            }
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>
            Cancel
          </Button>
          <Button onClick={handleSaveTask} variant="contained" color="primary">
            {selectedTask ? 'Update' : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Confirmation Dialog */}
      <Dialog
        open={confirmDialog.open}
        onClose={() => setConfirmDialog({ ...confirmDialog, open: false })}
      >
        <DialogTitle>{confirmDialog.title}</DialogTitle>
        <DialogContent>
          <DialogContentText>
            {confirmDialog.message}
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmDialog({ ...confirmDialog, open: false })}>
            Cancel
          </Button>
          <Button onClick={confirmDialog.onConfirm} color="error" variant="contained" autoFocus>
            Confirm
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default TaskManager;
