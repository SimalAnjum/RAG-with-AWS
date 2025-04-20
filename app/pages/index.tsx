'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Gavel } from 'lucide-react';
import {
  PaperClipIcon,
  PaperAirplaneIcon,
  DocumentTextIcon,
  XMarkIcon,
  MicrophoneIcon,
} from '@heroicons/react/24/outline';

export type DocumentFile = {
  id: string;
  name: string;
  size: number;
  type: string;
  content?: string;
};

type RetrievedChunk = {
  text: string;
  score?: number;
  [key: string]: any;
};

type Message = {
  id: string;
  content: string;
  isUser: boolean;
  timestamp: Date;
  context?: RetrievedChunk[];
};

const ChatbotPage = () => {
  const [messages, setMessages] = useState<Message[]>([{
    id: '1',
    content: 'Hello! Upload a document and ask a question about it. How can I help you today?',
    isUser: false,
    timestamp: new Date(),
  }]);

  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [activeContext, setActiveContext] = useState<RetrievedChunk[] | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = () => {
    if (inputMessage.trim() === '') return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      isUser: true,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    const formData = new FormData();
    formData.append('query', inputMessage);
    if (uploadedFiles[0]) {
      formData.append('file', uploadedFiles[0]);
    }

    fetch('http://localhost:8000/rag', {
      method: 'POST',
      body: formData,
    })
      .then((res) => res.json())
      .then((data) => {
        const botResponse: Message = {
          id: Date.now().toString(),
          content: data.answer || 'No answer returned.',
          isUser: false,
          timestamp: new Date(),
          context: data.context || [],
        };
        setMessages((prev) => [...prev, botResponse]);
      })
      .catch((error) => {
        const errorMsg: Message = {
          id: Date.now().toString(),
          content: `⚠️ Error: ${error.message}`,
          isUser: false,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, errorMsg]);
      })
      .finally(() => setIsLoading(false));
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setUploadedFiles([file]);

      const fileMessage: Message = {
        id: Date.now().toString(),
        content: `File uploaded: ${file.name}`,
        isUser: false,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, fileMessage]);
    }
  };

  const removeFile = () => setUploadedFiles([]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex h-screen">
      {/* Left Panel - Chat */}
      <div className="flex flex-col flex-1 bg-gradient-to-br from-gray-900 to-gray-700 text-white">
        <div className="bg-gray-800 p-4 shadow-md flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="bg-blue-600 h-10 w-10 rounded-lg flex items-center justify-center">
              <DocumentTextIcon className="h-6 w-6" />
            </div>
            <h1 className="text-xl font-bold">RAG Chatbot</h1>
          </div>
          <div className="bg-blue-600 rounded-full px-4 py-1 text-sm">Document Assistant</div>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((message) => (
            <div key={message.id} className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-3/4 rounded-lg px-4 py-2 ${message.isUser ? 'bg-blue-600' : 'bg-gray-700'}`}>
                <p className="whitespace-pre-wrap">{message.content}</p>
                {!message.isUser && message.context && message.context.length > 0 && (
                  <button
                    onClick={() => setActiveContext(message.context ?? [])}
                    className="text-xs mt-2 text-blue-400 hover:underline flex items-center space-x-1"
                  >
                    <DocumentTextIcon className="h-4 w-4" />
                    <span>Show context</span>
                  </button>
                )}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-gray-700 rounded-lg px-4 py-2 flex items-center space-x-2">
                <Gavel className="w-8 h-8 animate-bounce text-[#1A1F2C]" />
                <span>Please wait I am getting the answer for you...</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="bg-gray-800 p-4 border-t border-gray-700">
          <div className="flex items-center space-x-2">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              className="hidden"
              accept=".pdf,.txt"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="bg-gray-700 rounded-full p-2 hover:bg-gray-600"
              title="Upload Document"
            >
              <PaperClipIcon className="h-5 w-5" />
            </button>
            <div className="flex-1 bg-gray-700 rounded-lg px-4 py-2 flex items-center">
              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask me about your document..."
                className="bg-transparent border-none outline-none resize-none w-full max-h-32 text-white"
                rows={1}
              />
              <button className="text-gray-400 hover:text-white ml-2">
                <MicrophoneIcon className="h-5 w-5" />
              </button>
            </div>
            <button
              onClick={handleSendMessage}
              disabled={inputMessage.trim() === '' || isLoading}
              className={`rounded-full p-2 ${inputMessage.trim() === '' || isLoading ? 'bg-gray-600 text-gray-400' : 'bg-blue-600 text-white hover:bg-blue-500'} transition`}
            >
              <PaperAirplaneIcon className="h-5 w-5" />
            </button>
          </div>
        </div>
      </div>

      {/* Right Panel - Context */}
      {activeContext && (
        <div className="w-96 bg-gray-900 border-l border-gray-700 p-4 overflow-y-auto">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-blue-400">Context</h2>
            <button onClick={() => setActiveContext(null)} className="text-gray-400 hover:text-white">
              <XMarkIcon className="h-5 w-5" />
            </button>
          </div>
          {activeContext.map((chunk, idx) => (
            <div key={idx} className="mb-4 p-2 border border-gray-600 rounded-md bg-gray-800">
              <p className="text-gray-300 whitespace-pre-wrap">{chunk.text}</p>
              <p className="text-xs text-gray-400 mt-1">🔍 Relevance Score: {(chunk.score || 0).toFixed(2)}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ChatbotPage;
