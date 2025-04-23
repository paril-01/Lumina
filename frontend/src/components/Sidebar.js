import React, { useContext } from 'react';
import { 
  Drawer, 
  List, 
  ListItem, 
  ListItemIcon, 
  ListItemText, 
  Divider, 
  Toolbar, 
  Box, 
  Typography, 
  Avatar, 
  ListItemButton 
} from '@mui/material';
import { styled, alpha } from '@mui/material/styles';
import { useNavigate, useLocation } from 'react-router-dom';
import DashboardIcon from '@mui/icons-material/Dashboard';
import SettingsIcon from '@mui/icons-material/Settings';
import TaskIcon from '@mui/icons-material/AssignmentTurnedIn';
import ActivityIcon from '@mui/icons-material/Timeline';
import LearningIcon from '@mui/icons-material/Psychology';
import PrivacyIcon from '@mui/icons-material/Security';
import ChatIcon from '@mui/icons-material/Chat';
import EmojiObjectsIcon from '@mui/icons-material/EmojiObjects';
import ThemeContext from '../context/ThemeContext';
import AuthContext from '../context/AuthContext';

const drawerWidth = 240;

// Styled components
const StyledDrawer = styled(Drawer)(({ theme }) => ({
  '& .MuiDrawer-paper': {
    borderRight: 'none',
    backgroundColor: alpha(theme.palette.background.sidebar, 0.95),
    backdropFilter: 'blur(10px)',
  },
}));

const StyledListItemButton = styled(ListItemButton)(({ theme }) => ({
  borderRadius: 12,
  marginBottom: theme.spacing(0.5),
  '&.Mui-selected': {
    backgroundColor: alpha(theme.palette.primary.main, 0.1),
    '&:hover': {
      backgroundColor: alpha(theme.palette.primary.main, 0.15),
    },
    '& .MuiListItemIcon-root': {
      color: theme.palette.primary.main,
    },
    '& .MuiListItemText-primary': {
      fontWeight: 600,
      color: theme.palette.primary.main,
    },
  },
  '&:hover': {
    backgroundColor: alpha(theme.palette.primary.main, 0.08),
  },
}));

function Sidebar({ open }) {
  const navigate = useNavigate();
  const location = useLocation();
  const { mode } = useContext(ThemeContext);
  const { user } = useContext(AuthContext);

  // Menu items
  const menuItems = [
    { text: 'Chat', icon: <ChatIcon />, path: '/' },
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/dashboard' },
    { text: 'Task Automation', icon: <TaskIcon />, path: '/tasks' },
    { text: 'Activity Monitor', icon: <ActivityIcon />, path: '/activity' },
    { text: 'Learning Progress', icon: <LearningIcon />, path: '/learning' },
    { divider: true },
    { text: 'Settings', icon: <SettingsIcon />, path: '/settings' },
    { text: 'Privacy Policy', icon: <PrivacyIcon />, path: '/privacy' },
  ];

  return (
    <StyledDrawer
      variant="permanent"
      open={open}
      sx={{
        width: open ? drawerWidth : 64,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: open ? drawerWidth : 64,
          boxSizing: 'border-box',
          overflowX: 'hidden',
          transition: theme => theme.transitions.create('width', {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.enteringScreen,
          }),
        },
      }}
    >
      <Toolbar sx={{ height: 64 }} />
      
      {/* User info section when sidebar is expanded */}
      {open && (
        <Box sx={{ p: 2, mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <Avatar sx={{ width: 40, height: 40, bgcolor: 'primary.main', mr: 1.5 }}>
              <EmojiObjectsIcon />
            </Avatar>
            <Box>
              <Typography variant="body1" fontWeight="600" noWrap>
                {user?.username || 'User'}
              </Typography>
              <Typography variant="caption" color="text.secondary" noWrap>
                Personal AI Assistant
              </Typography>
            </Box>
          </Box>
        </Box>
      )}
      
      <Box sx={{ px: open ? 2 : 1 }}>
        <List sx={{ p: 0 }}>
          {menuItems.map((item, index) => (
            item.divider ? (
              <Divider key={`divider-${index}`} sx={{ my: 1.5 }} />
            ) : (
              <ListItem key={item.text} disablePadding sx={{ mb: 0.5 }}>
                <StyledListItemButton
                  onClick={() => navigate(item.path)}
                  selected={location.pathname === item.path}
                  sx={{
                    justifyContent: open ? 'initial' : 'center',
                    px: open ? 2 : 1,
                    py: 1.2,
                  }}
                >
                  <ListItemIcon 
                    sx={{ 
                      minWidth: 0, 
                      mr: open ? 2 : 0,
                      color: 'text.secondary',
                      justifyContent: 'center',
                    }}
                  >
                    {item.icon}
                  </ListItemIcon>
                  {open && (
                    <ListItemText 
                      primary={item.text} 
                      primaryTypographyProps={{ 
                        fontSize: 14,
                        fontWeight: location.pathname === item.path ? 600 : 500 
                      }} 
                    />
                  )}
                </StyledListItemButton>
              </ListItem>
            )
          ))}
        </List>
      </Box>
      
      {/* App version */}
      {open && (
        <Box sx={{ p: 2, mt: 'auto', mb: 2 }}>
          <Typography variant="caption" color="text.secondary">
            Lumina v1.0.0
          </Typography>
        </Box>
      )}
    </StyledDrawer>
  );
}

export default Sidebar;
