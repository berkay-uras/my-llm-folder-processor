import React, { useState } from 'react';
import axios from 'axios';
import './index.css';

// Material-UI Components
import {
  Container,
  Box,
  Typography,
  Button,
  TextField,
  CircularProgress,
  Alert,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Paper,
} from '@mui/material';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import DescriptionIcon from '@mui/icons-material/Description';
import SendIcon from '@mui/icons-material/Send';
import QuestionAnswerIcon from '@mui/icons-material/QuestionAnswer';

function App() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [llmResponse, setLlmResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [question, setQuestion] = useState('');
  const [llmAnswer, setLlmAnswer] = useState('');
  const [isProcessingQuestion, setIsProcessingQuestion] = useState(false);

  // Manages the folder selection event
  const handleFolderSelect = (event) => {
    const files = Array.from(event.target.files);
    setSelectedFiles(files);
    setError('');
    setSuccess('');
    setLlmResponse('');
    setQuestion('');
    setLlmAnswer('');
    console.log("Selected files:", files.map(f => f.name));
  };

  // When "Upload Files and Sync" button is clicked
  const handleSubmit = async () => {
    if (selectedFiles.length === 0) {
      setError('Please select a folder to process.');
      return;
    }

    setLoading(true);
    setLlmResponse('');
    setError('');
    setSuccess('');
    setQuestion('');
    setLlmAnswer('');

    const formData = new FormData();
    selectedFiles.forEach(file => {
      formData.append('files', file);
    });

    try {
      // Endpoint adını değiştiriyoruz
      const response = await axios.post('http://localhost:8000/sync-and-process/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setLlmResponse(response.data.llm_response);
      setSuccess(`Success: ${response.data.message}`);
    } catch (err) {
      console.error('File sync or LLM error:', err);
      if (err.response && err.response.data && err.response.data.detail) {
        setError(`Error: ${err.response.data.detail}`);
      } else {
        setError('An error occurred while syncing files or getting LLM response.');
      }
    } finally {
      setLoading(false);
    }
  };

  // Function to ask LLM a question
  const handleAskLlm = async () => {
    if (!question.trim()) {
      setError('Please enter a question.');
      return;
    }
    if (!llmResponse) {
      setError('You need to sync files from the folder first.');
      return;
    }

    setIsProcessingQuestion(true);
    setLlmAnswer('');
    setError('');

    try {
      const response = await axios.post('http://localhost:8000/ask-llm/', { question: question });
      setLlmAnswer(response.data.llm_answer);
      setSuccess('LLM response successfully received!');
    } catch (err) {
      console.error('LLM question error:', err);
      if (err.response && err.response.data && err.response.data.detail) {
        setError(`Error: ${err.response.data.detail}`);
      } else {
        setError('An error occurred while getting LLM response.');
      }
    } finally {
      setIsProcessingQuestion(false);
    }
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Paper elevation={3} sx={{ p: 4, borderRadius: 3, boxShadow: '0px 10px 30px rgba(0, 0, 0, 0.1)' }}>
        <Typography variant="h3" component="h1" align="center" gutterBottom sx={{ fontWeight: 700, color: '#1a202c' }}>
          <FolderOpenIcon sx={{ fontSize: 40, verticalAlign: 'middle', mr: 1, color: '#3f51b5' }} />
          Smart Folder Processor
        </Typography>
        <Typography variant="body1" align="center" color="text.secondary" sx={{ mb: 4, lineHeight: 1.7 }}>
          Select a folder containing the files you want the LLM to interpret.
          All supported files (TXT, PDF, DOCX, XLSX) in the selected folder will be uploaded and synchronized with the knowledge base.
        </Typography>

        <Box sx={{ mb: 4, p: 3, backgroundColor: '#e8f0fe', borderRadius: 2, border: '1px solid #c5d8ff' }}>
          <Typography variant="h6" component="label" htmlFor="folder-input" sx={{ mb: 2, display: 'block', color: '#3f51b5', fontWeight: 600 }}>
            1. Select Folder:
          </Typography>
          <Button
            variant="contained"
            component="label"
            startIcon={<UploadFileIcon />}
            sx={{
              backgroundColor: '#3f51b5',
              '&:hover': { backgroundColor: '#303f9f' },
              color: 'white',
              py: 1.5,
              px: 3,
              borderRadius: 2,
              boxShadow: '0px 4px 15px rgba(0, 0, 0, 0.1)',
              transition: 'transform 0.2s',
              '&:active': { transform: 'scale(0.98)' }
            }}
          >
            Select Folder
            <input
              type="file"
              id="folder-input"
              webkitdirectory="true"
              directory="true"
              onChange={handleFolderSelect}
              hidden
            />
          </Button>
          {selectedFiles.length > 0 && (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              Number of selected files: <Typography component="span" sx={{ fontWeight: 600, color: '#3f51b5' }}>{selectedFiles.length}</Typography>
              <List dense sx={{ maxHeight: 150, overflowY: 'auto', mt: 1, border: '1px solid #e0e0e0', borderRadius: 1, p: 0.5 }}>
                {selectedFiles.map((file, index) => (
                  <ListItem key={index} sx={{ py: 0.5 }}>
                    <ListItemIcon sx={{ minWidth: 30 }}>
                      <DescriptionIcon fontSize="small" color="action" />
                    </ListItemIcon>
                    <ListItemText primary={file.name} primaryTypographyProps={{ fontSize: '0.85rem' }} />
                  </ListItem>
                ))}
              </List>
            </Typography>
          )}
        </Box>

        <Button
          variant="contained"
          onClick={handleSubmit}
          disabled={loading || selectedFiles.length === 0}
          fullWidth
          size="large"
          startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <SendIcon />}
          sx={{
            backgroundColor: '#4caf50',
            '&:hover': { backgroundColor: '#388e3c' },
            color: 'white',
            py: 1.5,
            px: 3,
            borderRadius: 2,
            boxShadow: '0px 5px 20px rgba(0, 0, 0, 0.15)',
            transition: 'transform 0.2s',
            '&:active': { transform: 'scale(0.98)' },
            fontSize: '1.1rem',
            fontWeight: 600,
          }}
        >
          {loading ? 'Synchronizing...' : 'Sync Files with Knowledge Base'}
        </Button>

        {error && (
          <Alert severity="error" icon={<ErrorOutlineIcon fontSize="inherit" />} sx={{ mt: 3, borderRadius: 2 }}>
            <Typography variant="body1" sx={{ fontWeight: 500 }}>Error!</Typography>
            <Typography variant="body2">{error}</Typography>
          </Alert>
        )}

        {success && (
          <Alert severity="success" icon={<CheckCircleOutlineIcon fontSize="inherit" />} sx={{ mt: 3, borderRadius: 2 }}>
            <Typography variant="body1" sx={{ fontWeight: 500 }}>Success!</Typography>
            <Typography variant="body2">{success}</Typography>
          </Alert>
        )}

        {llmResponse && (
          <Box sx={{ mt: 4, p: 3, backgroundColor: '#f5f5f5', borderRadius: 2, border: '1px solid #e0e0e0', boxShadow: 'inset 0px 2px 5px rgba(0, 0, 0, 0.05)' }}>
            <Typography variant="h5" component="h2" gutterBottom sx={{ fontWeight: 600, color: '#673ab7' }}>
              <span role="img" aria-label="sparkle">-</span> LLM Status:
            </Typography>
            <Paper variant="outlined" sx={{ p: 2, borderRadius: 1, backgroundColor: 'white', maxHeight: 300, overflowY: 'auto' }}>
              <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', lineHeight: 1.6, color: '#424242' }}>{llmResponse}</Typography>
            </Paper>

            {/* Question Asking Section */}
            <Box sx={{ mt: 4, p: 3, backgroundColor: '#e0f2f7', borderRadius: 2, border: '1px solid #b2ebf2' }}>
              <Typography variant="h6" sx={{ mb: 2, color: '#006064', fontWeight: 600 }}>
                2. Ask the LLM a Question:
              </Typography>
              <TextField
                fullWidth
                label="Type your question here..."
                variant="outlined"
                multiline
                rows={3}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                sx={{ mb: 2, '& .MuiOutlinedInput-root': { backgroundColor: 'white' } }}
              />
              <Button
                variant="contained"
                onClick={handleAskLlm}
                disabled={isProcessingQuestion || !question.trim()}
                fullWidth
                size="medium"
                startIcon={isProcessingQuestion ? <CircularProgress size={20} color="inherit" /> : <QuestionAnswerIcon />}
                sx={{
                  backgroundColor: '#00bcd4',
                  '&:hover': { backgroundColor: '#00838f' },
                  color: 'white',
                  py: 1.2,
                  px: 3,
                  borderRadius: 2,
                  boxShadow: '0px 3px 10px rgba(0, 0, 0, 0.1)',
                  transition: 'transform 0.2s',
                  '&:active': { transform: 'scale(0.98)' },
                  fontWeight: 600,
                }}
              >
                {isProcessingQuestion ? 'Generating Response...' : 'Ask LLM'}
              </Button>
            </Box>

            {llmAnswer && (
              <Box sx={{ mt: 4, p: 3, backgroundColor: '#e8eaf6', borderRadius: 2, border: '1px solid #c5cae9' }}>
                <Typography variant="h6" component="h2" gutterBottom sx={{ fontWeight: 600, color: '#3f51b5' }}>
                  <QuestionAnswerIcon sx={{ verticalAlign: 'middle', mr: 1 }} /> LLM Answer:
                </Typography>
                <Paper variant="outlined" sx={{ p: 2, borderRadius: 1, backgroundColor: 'white', maxHeight: 250, overflowY: 'auto' }}>
                  <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', lineHeight: 1.6, color: '#424242' }}>{llmAnswer}</Typography>
                </Paper>
              </Box>
            )}
          </Box>
        )}
      </Paper>
    </Container>
  );
}

export default App;