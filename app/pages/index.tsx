// // File: app/page.tsx
// 'use client';

// import React, { useState, useRef, useEffect } from 'react';
// import { 
//   PaperClipIcon, 
//   PaperAirplaneIcon, 
//   DocumentTextIcon,
//   XMarkIcon,
//   ArrowPathIcon,
//   MicrophoneIcon
// } from '@heroicons/react/24/outline';
// import Image from 'next/image';

// export type DocumentFile = {
//   id: string;
//   name: string;
//   size: number;
//   type: string;
//   content?: string; // This would store extracted text content
// };

// export type ChatState = {
//   messages: Message[];
//   documents: DocumentFile[];
//   isLoading: boolean;
//   error: string | null;
// };

// type Message = {
//   id: string;
//   content: string;
//   isUser: boolean;
//   timestamp: Date;
// };

// const ChatbotPage = () => {
//   const [messages, setMessages] = useState<Message[]>([
//     {
//       id: '1',
//       content: 'Hello! Upload documents using the clip icon, and I can answer questions based on their content. How can I help you today?',
//       isUser: false,
//       timestamp: new Date()
//     }
//   ]);
  
//   const [inputMessage, setInputMessage] = useState('');
//   const [isLoading, setIsLoading] = useState(false);
//   const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
//   const fileInputRef = useRef<HTMLInputElement>(null);
//   const messagesEndRef = useRef<HTMLDivElement>(null);

//   const handleSendMessage = () => {
//     if (inputMessage.trim() === '') return;
    
//     // Add user message
//     const userMessage: Message = {
//       id: Date.now().toString(),
//       content: inputMessage,
//       isUser: true,
//       timestamp: new Date()
//     };
    
//     setMessages(prev => [...prev, userMessage]);
//     setInputMessage('');
//     setIsLoading(true);
    
//     // Simulate response (in a real app, this would call your RAG backend)
//     setTimeout(() => {
//       const botResponse: Message = {
//         id: (Date.now() + 1).toString(),
//         content: `I've processed your question: "${inputMessage}". ${uploadedFiles.length > 0 ? 'Based on your uploaded documents, here is my answer...' : 'Please upload documents for more relevant answers.'}`,
//         isUser: false,
//         timestamp: new Date()
//       };
      
//       setMessages(prev => [...prev, botResponse]);
//       setIsLoading(false);
//     }, 1500);
//   };

//   const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
//     if (e.target.files && e.target.files.length > 0) {
//       const newFiles = Array.from(e.target.files);
//       setUploadedFiles(prev => [...prev, ...newFiles]);
      
//       // Add system message about uploaded files
//       const fileMessage: Message = {
//         id: Date.now().toString(),
//         content: `Files uploaded: ${newFiles.map(file => file.name).join(', ')}`,
//         isUser: false,
//         timestamp: new Date()
//       };
      
//       setMessages(prev => [...prev, fileMessage]);
//     }
//   };

//   const removeFile = (index: number) => {
//     setUploadedFiles(prev => prev.filter((_, i) => i !== index));
//   };

//   const handleKeyDown = (e: React.KeyboardEvent) => {
//     if (e.key === 'Enter' && !e.shiftKey) {
//       e.preventDefault();
//       handleSendMessage();
//     }
//   };

//   // Auto-scroll to bottom when new messages arrive
//   useEffect(() => {
//     messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
//   }, [messages]);

//   return (
//     <div className="flex flex-col h-screen bg-gray-900 text-white">
//       {/* Header */}
//       <div className="bg-gray-800 p-4 shadow-md">
//         <div className="flex items-center justify-between">
//           <div className="flex items-center space-x-3">
//             <div className="bg-blue-600 h-10 w-10 rounded-lg flex items-center justify-center">
//               <DocumentTextIcon className="h-6 w-6" />
//             </div>
//             <h1 className="text-xl font-bold">Legal RAG Chatbot</h1>
//           </div>
//           <div className="bg-blue-600 rounded-full px-4 py-1 text-sm">
//             Document Assistant
//           </div>
//         </div>
//       </div>

//       {/* Uploaded Files Section */}
//       {uploadedFiles.length > 0 && (
//         <div className="bg-gray-800 p-2 border-t border-gray-700">
//           <div className="flex overflow-x-auto space-x-2 py-1">
//             {uploadedFiles.map((file, index) => (
//               <div key={index} className="flex items-center bg-gray-700 rounded-lg px-3 py-1 text-sm whitespace-nowrap">
//                 <DocumentTextIcon className="h-4 w-4 mr-2" />
//                 <span className="truncate max-w-xs">{file.name}</span>
//                 <button 
//                   onClick={() => removeFile(index)} 
//                   className="ml-2 text-gray-400 hover:text-white"
//                 >
//                   <XMarkIcon className="h-4 w-4" />
//                 </button>
//               </div>
//             ))}
//           </div>
//         </div>
//       )}

//       {/* Chat Messages */}
//       <div className="flex-1 overflow-y-auto p-4 space-y-4">
//         {messages.map((message) => (
//           <div 
//             key={message.id} 
//             className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}
//           >
//             <div 
//               className={`max-w-3/4 rounded-lg px-4 py-2 ${
//                 message.isUser 
//                   ? 'bg-blue-600 text-white' 
//                   : 'bg-gray-700 text-white'
//               }`}
//             >
//               <p className="whitespace-pre-wrap">{message.content}</p>
//               <p className="text-xs opacity-70 mt-1">
//                 {message.timestamp.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
//               </p>
//             </div>
//           </div>
//         ))}
//         {isLoading && (
//           <div className="flex justify-start">
//             <div className="bg-gray-700 rounded-lg px-4 py-2 flex items-center space-x-2">
//               <ArrowPathIcon className="h-5 w-5 animate-spin" />
//               <span>Processing...</span>
//             </div>
//           </div>
//         )}
//         <div ref={messagesEndRef} />
//       </div>

//       {/* Input Area */}
//       <div className="bg-gray-800 p-4 border-t border-gray-700">
//         <div className="flex items-center space-x-2">
//           <input 
//             type="file" 
//             ref={fileInputRef} 
//             onChange={handleFileUpload} 
//             className="hidden" 
//             multiple 
//           />
//           <button 
//             onClick={() => fileInputRef.current?.click()} 
//             className="bg-gray-700 rounded-full p-2 hover:bg-gray-600 transition"
//             title="Upload Documents"
//           >
//             <PaperClipIcon className="h-5 w-5" />
//           </button>
//           <div className="flex-1 bg-gray-700 rounded-lg px-4 py-2 flex items-center">
//             <textarea
//               value={inputMessage}
//               onChange={(e) => setInputMessage(e.target.value)}
//               onKeyDown={handleKeyDown}
//               placeholder="Ask me about your documents..."
//               className="bg-transparent border-none outline-none resize-none w-full max-h-32 text-white"
//               rows={1}
//             />
//             <button className="text-gray-400 hover:text-white ml-2">
//               <MicrophoneIcon className="h-5 w-5" />
//             </button>
//           </div>
//           <button 
//             onClick={handleSendMessage} 
//             disabled={inputMessage.trim() === '' || isLoading}
//             className={`rounded-full p-2 ${
//               inputMessage.trim() === '' || isLoading 
//                 ? 'bg-gray-600 text-gray-400' 
//                 : 'bg-blue-600 text-white hover:bg-blue-500'
//             } transition`}
//           >
//             <PaperAirplaneIcon className="h-5 w-5" />
//           </button>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default ChatbotPage;



// File: app/page.tsx
'use client';

import React, { useState, useRef, useEffect } from 'react';
import { 
  PaperClipIcon, 
  PaperAirplaneIcon, 
  DocumentTextIcon,
  XMarkIcon,
  ArrowPathIcon,
  MicrophoneIcon
} from '@heroicons/react/24/outline';
import Image from 'next/image';

export type DocumentFile = {
  id: string;
  name: string;
  size: number;
  type: string;
  content?: string;
};

type Message = {
  id: string;
  content: string;
  isUser: boolean;
  timestamp: Date;
};

const ChatbotPage = () => {
  const [messages, setMessages] = useState<Message[]>([{
    id: '1',
    content: 'Hello! Upload a document and ask a question about it. How can I help you today?',
    isUser: false,
    timestamp: new Date()
  }]);

  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const handleSendMessage = () => {
    if (inputMessage.trim() === '') return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      isUser: true,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    const formData = new FormData();
    formData.append("query", inputMessage);
    if (uploadedFiles[0]) {
      formData.append("file", uploadedFiles[0]);
    }

    fetch("http://localhost:8000/rag", {
      method: "POST",
      body: formData,
    })
      .then((res) => res.json())
      .then((data) => {
        const botResponse: Message = {
          id: Date.now().toString(),
          content: data.answer || "No answer returned.",
          isUser: false,
          timestamp: new Date()
        };
        setMessages((prev) => [...prev, botResponse]);
      })
      .catch((error) => {
        const errorMsg: Message = {
          id: Date.now().toString(),
          content: `⚠️ Error: ${error.message}`,
          isUser: false,
          timestamp: new Date()
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
        timestamp: new Date()
      };
      setMessages(prev => [...prev, fileMessage]);
    }
  };

  const removeFile = () => setUploadedFiles([]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-white">
      <div className="bg-gray-800 p-4 shadow-md flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="bg-blue-600 h-10 w-10 rounded-lg flex items-center justify-center">
            <DocumentTextIcon className="h-6 w-6" />
          </div>
          <h1 className="text-xl font-bold">Legal RAG Chatbot</h1>
        </div>
        <div className="bg-blue-600 rounded-full px-4 py-1 text-sm">Document Assistant</div>
      </div>

      {uploadedFiles.length > 0 && (
        <div className="bg-gray-800 p-2 border-t border-gray-700">
          <div className="flex items-center space-x-2">
            <DocumentTextIcon className="h-4 w-4" />
            <span className="truncate">{uploadedFiles[0].name}</span>
            <button onClick={removeFile} className="text-gray-400 hover:text-white">
              <XMarkIcon className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div key={message.id} className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-3/4 rounded-lg px-4 py-2 ${message.isUser ? 'bg-blue-600' : 'bg-gray-700'}`}>
              <p className="whitespace-pre-wrap">{message.content}</p>
              <p className="text-xs opacity-70 mt-1">
                {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </p>
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-gray-700 rounded-lg px-4 py-2 flex items-center space-x-2">
              <ArrowPathIcon className="h-5 w-5 animate-spin" />
              <span>Processing...</span>
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
  );
};

export default ChatbotPage;
