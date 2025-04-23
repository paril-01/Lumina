import React, { useState, useContext } from 'react';
import { AppBar, Toolbar, Typography, IconButton, Avatar, Menu, MenuItem, Box, Badge, Tooltip } from '@mui/material';
import { styled, alpha } from '@mui/material/styles';
import MenuIcon from '@mui/icons-material/Menu';
import NotificationsIcon from '@mui/icons-material/Notifications';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import DarkModeIcon from '@mui/icons-material/DarkMode';
import LightModeIcon from '@mui/icons-material/LightMode';
import EmojiObjectsIcon from '@mui/icons-material/EmojiObjects';
import AuthContext from '../context/AuthContext';
import ThemeContext from '../context/ThemeContext';
import { Link } from 'react-router-dom';
import { useNavigate } from 'react-router-dom';

// Styled components
const StyledAppBar = styled(AppBar)(({ theme }) => ({
  boxShadow: 'none',
  backdropFilter: 'blur(10px)',
  backgroundColor: alpha(theme.palette.background.paper, 0.8),
  borderBottom: `1px solid ${theme.palette.divider}`,
  zIndex: theme.zIndex.drawer + 1,
}));

const StyledBadge = styled(Badge)(({ theme }) => ({
  '& .MuiBadge-badge': {
    backgroundColor: theme.palette.secondary.main,
    color: theme.palette.secondary.contrastText,
    fontWeight: 'bold',
    fontSize: 10,
    height: 16,
    minWidth: 16,
    padding: '0 4px',
  },
}));

function Header({ toggleSidebar }) {
  const { logout } = useContext(AuthContext);
  // Fallback in case logout is not defined
  const safeLogout = typeof logout === 'function' ? logout : () => { console.warn('Logout function is not defined in AuthContext'); };
  const { mode, toggleTheme } = useContext(ThemeContext);
  const [anchorEl, setAnchorEl] = useState(null);
  const [notificationAnchorEl, setNotificationAnchorEl] = useState(null);
  const [notifications, setNotifications] = useState([{
    id: 1,
    message: 'Welcome to Lumina AI Assistant!', 
    read: false
  }]);
  
  const unreadCount = notifications.filter(n => !n.read).length;
  
  const handleMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleNotificationMenuOpen = (event) => {
    setNotificationAnchorEl(event.currentTarget);
    // Mark notifications as read when opened
    setNotifications(prev => 
      prev.map(notification => ({ ...notification, read: true }))
    );
  };

  const handleNotificationMenuClose = () => {
    setNotificationAnchorEl(null);
  };

  const handleLogout = () => {
    handleMenuClose();
    safeLogout();
  };


  const navigate = useNavigate();
  const handleSettings = () => {
    handleMenuClose();
    navigate('/settings');
  };

  return (
    <StyledAppBar position="fixed">
      <Toolbar sx={{ justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <IconButton
            edge="start"
            color="inherit"
            aria-label="toggle sidebar"
            onClick={toggleSidebar}
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Avatar 
              sx={{ 
                bgcolor: 'primary.main', 
                width: 32, 
                height: 32, 
                mr: 1,
                boxShadow: 1 
              }}
            >
              <EmojiObjectsIcon fontSize="small" />
            </Avatar>
            <Typography 
              variant="h6" 
              component={Link} 
              to="/"
              sx={{ 
                flexGrow: 1, 
                color: 'text.primary',
                textDecoration: 'none',
                fontWeight: 600,
                letterSpacing: '-0.5px'
              }}
            >
              Lumina
            </Typography>
          </Box>
        </Box>
        
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Tooltip title={`Switch to ${mode === 'light' ? 'dark' : 'light'} mode`}>
            <IconButton onClick={toggleTheme} color="inherit" sx={{ mx: 1 }}>
              {mode === 'light' ? <DarkModeIcon /> : <LightModeIcon />}
            </IconButton>
          </Tooltip>
          
          <Tooltip title="Notifications">
            <IconButton color="inherit" onClick={handleNotificationMenuOpen} sx={{ mx: 1 }}>
              <StyledBadge badgeContent={unreadCount} color="secondary">
                <NotificationsIcon />
              </StyledBadge>
            </IconButton>
          </Tooltip>

          <Tooltip title="Account settings">
            <IconButton 
              color="inherit" 
              onClick={handleMenuOpen}
              sx={{ 
                ml: 1,
                border: 1,
                borderColor: 'divider',
                borderRadius: 2,
                p: 0.5,
              }}
            >
              <AccountCircleIcon />
            </IconButton>
          </Tooltip>
        </Box>

        <Menu
          id="menu-appbar"
          anchorEl={anchorEl}
          anchorOrigin={{
            vertical: 'bottom',
            horizontal: 'right',
          }}
          keepMounted
          transformOrigin={{
            vertical: 'top',
            horizontal: 'right',
          }}
          open={Boolean(anchorEl)}
          onClose={handleMenuClose}
          PaperProps={{
            elevation: 2,
            sx: {
              borderRadius: 2,
              minWidth: 180,
              boxShadow: (theme) => theme.palette.mode === 'light'
                ? '0 4px 20px rgba(0,0,0,0.08)'
                : '0 4px 20px rgba(0,0,0,0.25)',
            }
          }}
        >
          <MenuItem onClick={handleMenuClose} component={Link} to="/dashboard">
            Dashboard
          </MenuItem>
          <MenuItem onClick={handleSettings}>Settings</MenuItem>
          <MenuItem onClick={handleLogout}>Logout</MenuItem>
        </Menu>
        
        <Menu
          id="notification-menu"
          anchorEl={notificationAnchorEl}
          anchorOrigin={{
            vertical: 'bottom',
            horizontal: 'right',
          }}
          keepMounted
          transformOrigin={{
            vertical: 'top',
            horizontal: 'right',
          }}
          open={Boolean(notificationAnchorEl)}
          onClose={handleNotificationMenuClose}
          PaperProps={{
            elevation: 2,
            sx: {
              borderRadius: 2,
              minWidth: 280,
              maxWidth: 320,
              boxShadow: (theme) => theme.palette.mode === 'light'
                ? '0 4px 20px rgba(0,0,0,0.08)'
                : '0 4px 20px rgba(0,0,0,0.25)',
            }
          }}
        >
          {notifications.length > 0 ? (
            notifications.map(notification => (
              <MenuItem 
                key={notification.id} 
                onClick={handleNotificationMenuClose}
                sx={{
                  borderLeft: notification.read ? 'none' : '3px solid',
                  borderColor: 'secondary.main',
                  py: 1.5,
                }}
              >
                <Typography variant="body2">{notification.message}</Typography>
              </MenuItem>
            ))
          ) : (
            <MenuItem>
              <Typography variant="body2">No notifications</Typography>
            </MenuItem>
          )}
        </Menu>
      </Toolbar>
    </StyledAppBar>
  );
}

export default Header;
