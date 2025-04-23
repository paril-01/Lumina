import React, { useState, useRef, useEffect, useContext } from 'react';
import { Box, TextField, Typography, IconButton, Avatar, Paper, Divider, CircularProgress } from '@mui/material';
import { styled, alpha } from '@mui/material/styles';
import SendIcon from '@mui/icons-material/Send';
import MicIcon from '@mui/icons-material/Mic';
import AttachFileIcon from '@mui/icons-material/AttachFile';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import EmojiObjectsIcon from '@mui/icons-material/EmojiObjects';
import ThemeContext from '../context/ThemeContext';
import AuthContext from '../context/AuthContext';
import axios from 'axios';

// Styled components for chat bubbles
const UserMessage = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  marginBottom: theme.spacing(2),
  maxWidth: '75%',
  borderRadius: '18px 18px 4px 18px',
  backgroundColor: alpha(theme.palette.primary.main, 0.1),
  color: theme.palette.text.primary,
  alignSelf: 'flex-end',
  wordWrap: 'break-word',
  position: 'relative',
}));

const AssistantMessage = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  marginBottom: theme.spacing(2),
  maxWidth: '75%',
  borderRadius: '18px 18px 18px 4px',
  backgroundColor: theme.palette.mode === 'light' 
    ? alpha(theme.palette.background.paper, 0.9)
    : alpha(theme.palette.background.paper, 0.6),
  color: theme.palette.text.primary,
  alignSelf: 'flex-start',
  boxShadow: theme.shadows[1],
  wordWrap: 'break-word',
  position: 'relative',
}));

const ChatTextField = styled(TextField)(({ theme }) => ({
  '& .MuiOutlinedInput-root': {
    borderRadius: 30,
    backgroundColor: theme.palette.mode === 'light' ? alpha(theme.palette.common.white, 0.9) : alpha(theme.palette.background.paper, 0.8),
    transition: theme.transitions.create(['border-color', 'box-shadow', 'background-color']),
    '&.Mui-focused': {
      boxShadow: `0 0 0 2px ${alpha(theme.palette.primary.main, 0.25)}`,
    },
  },
}));

const ChatInterface = () => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([
    { 
      id: 1, 
      sender: 'assistant', 
      content: "Hi, I'm Lumina, your personal AI assistant. I can help you with tasks, learn from your activities, and automate actions for you. How can I help you today?",
      timestamp: new Date().toISOString()
    }
  ]);
  const [loading, setLoading] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const messagesEndRef = useRef(null);
  const { mode } = useContext(ThemeContext);
  const { user } = useContext(AuthContext);

  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleInputChange = (e) => {
    setInput(e.target.value);
  };

  const sendMessage = async () => {
    if (input.trim() === '') return;
    
    const userMessage = {
      id: Date.now(),
      sender: 'user',
      content: input,
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsThinking(true);
    
    // Simulate AI response for now
    try {
      // Uncomment and modify when backend is ready:
      // const response = await axios.post('/api/chat', {
      //   message: input,
      //   userId: user.id
      // });
      
      // Simulated response delay
      setTimeout(() => {
        const assistantMessage = {
          id: Date.now() + 1,
          sender: 'assistant',
          content: `I've processed your request: "${input}". As a personal AI assistant, I'm continuously learning to provide better assistance. Your activity data is safely stored on your device.`,
          timestamp: new Date().toISOString()
        };
        
        setMessages(prev => [...prev, assistantMessage]);
        setIsThinking(false);
      }, 1500);
      
    } catch (error) {
      console.error('Error sending message:', error);
      setIsThinking(false);
      
      const errorMessage = {
        id: Date.now() + 1,
        sender: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        timestamp: new Date().toISOString(),
        isError: true
      };
      
      setMessages(prev => [...prev, errorMessage]);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Chat header */}
      <Box
        sx={{
          p: 2,
          borderBottom: 1,
          borderColor: 'divider',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          backdropFilter: 'blur(10px)',
          backgroundColor: (theme) => alpha(theme.palette.background.paper, 0.8),
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Avatar sx={{ bgcolor: 'primary.main', mr: 1.5 }}>
            <EmojiObjectsIcon />
          </Avatar>
          <Typography variant="h6">Lumina</Typography>
        </Box>
        <IconButton>
          <MoreVertIcon />
        </IconButton>
      </Box>

      {/* Messages container */}
      <Box
        sx={{
          p: 2,
          flexGrow: 1,
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          backgroundColor: (theme) => 
            theme.palette.mode === 'light' 
              ? 'rgba(245, 245, 250, 0.8)' 
              : alpha(theme.palette.background.default, 0.6),
          backgroundImage: (theme) => 
            theme.palette.mode === 'light' 
              ? 'radial-gradient(circle at 50% 50%, rgba(200, 200, 255, 0.1) 0%, rgba(255, 255, 255, 0) 100%)' 
              : 'radial-gradient(circle at 50% 50%, rgba(30, 30, 60, 0.2) 0%, rgba(10, 10, 30, 0) 100%)',
        }}
      >
        {messages.map((message) => (
          message.sender === 'user' ? (
            <UserMessage key={message.id} elevation={0}>
              <Typography variant="body1">{message.content}</Typography>
              <Typography 
                variant="caption" 
                color="textSecondary"
                sx={{ 
                  display: 'block', 
                  textAlign: 'right',
                  mt: 1,
                  opacity: 0.6
                }}
              >
                {formatTimestamp(message.timestamp)}
              </Typography>
            </UserMessage>
          ) : (
            <AssistantMessage key={message.id} elevation={1}>
              <Typography variant="body1">{message.content}</Typography>
              <Typography 
                variant="caption" 
                color="textSecondary"
                sx={{ 
                  display: 'block', 
                  mt: 1,
                  opacity: 0.6
                }}
              >
                {formatTimestamp(message.timestamp)}
              </Typography>
            </AssistantMessage>
          )
        ))}
        
        {/* Thinking indicator */}
        {isThinking && (
          <AssistantMessage elevation={1} sx={{ display: 'flex', alignItems: 'center', p: 1.5 }}>
            <CircularProgress size={20} color="primary" sx={{ mr: 1 }} />
            <Typography variant="body2">Thinking...</Typography>
          </AssistantMessage>
        )}
        
        <div ref={messagesEndRef} />
      </Box>

      {/* Input area */}
      <Box
        sx={{
          p: 2,
          borderTop: 1,
          borderColor: 'divider',
          backgroundColor: (theme) => alpha(theme.palette.background.paper, 0.8),
          backdropFilter: 'blur(10px)',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <IconButton sx={{ mr: 1 }}>
            <AttachFileIcon />
          </IconButton>
          
          <ChatTextField
            fullWidth
            placeholder="Message Lumina..."
            variant="outlined"
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            disabled={loading}
            multiline
            maxRows={4}
            InputProps={{
              endAdornment: (
                <Box sx={{ display: 'flex' }}>
                  <IconButton>
                    <MicIcon />
                  </IconButton>
                  <IconButton 
                    color="primary" 
                    onClick={sendMessage} 
                    disabled={input.trim() === '' || loading}
                  >
                    <SendIcon />
                  </IconButton>
                </Box>
              ),
            }}
          />
        </Box>
      </Box>
    </Box>
  );
};

export default ChatInterface;
