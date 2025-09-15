import { Component, AfterViewChecked, ElementRef, ViewChild } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { NgIf, NgForOf, NgClass, DatePipe } from '@angular/common';
import { EmojiPipe } from 'src/app/emoji.pipe';              // ✅ standalone pipe
import { UserAuthService } from 'src/app/services/UserAuthService';

// ----------------------------
// TYPES
// ----------------------------
type Message = {
  role: 'user' | 'assistant';
  content: string;
  provenance?: string[];
  dates_found?: string[];
  times_found?: string[];
  warnings?: string[];
  contact_hr?: boolean;
  fileName?: string;
  fileType?: string;
  fileData?: Blob;
  downloadLink?: string;
  timestamp?: Date;
};

type Chat = {
  id: number;
  title: string;
  messages: Message[];
};

// ----------------------------
// COMPONENT
// ----------------------------
@Component({
  selector: 'app-chat-groq',
  templateUrl: './chat-groq.html',
  styleUrls: ['./chat-groq.scss'],
  standalone: true,
  imports: [NgIf, NgForOf, NgClass, FormsModule, DatePipe, EmojiPipe],
})
export class ChatGroq implements AfterViewChecked {
  @ViewChild('messagesContainer', { static: false }) private messagesContainer!: ElementRef;

  userInput = '';
  messages: Message[] = [];
  isLoading = false;
  selectedFile: File | null = null;

  chatHistory: Chat[] = [];
  currentChatId: number | null = null;

  constructor(private http: HttpClient, private authService: UserAuthService) {
    this.loadChatHistory();
  }

  ngAfterViewChecked() {
    this.scrollToBottom();
  }

  private scrollToBottom(): void {
    try {
      this.messagesContainer.nativeElement.scrollTop =
        this.messagesContainer.nativeElement.scrollHeight;
    } catch {}
  }

  // ----------------------------
  // UTILISATEUR CONNECTÉ
  // ----------------------------
  get currentUser() {
    return {
      id: this.authService.getId(),
      email: localStorage.getItem('email') || '',
      username: localStorage.getItem('username') || ''
    };
  }

  // ----------------------------
  // HISTORIQUE PAR UTILISATEUR
  // ----------------------------
  private getStorageKey(): string {
    return `chatHistory_${this.currentUser.id}`;
  }

  loadChatHistory() {
    const saved = localStorage.getItem(this.getStorageKey());
    if (saved) {
      this.chatHistory = JSON.parse(saved);
      if (this.chatHistory.length) {
        this.currentChatId = this.chatHistory[0].id;
        this.messages = this.chatHistory[0].messages;
      }
    } else {
      this.chatHistory = [];
      this.currentChatId = null;
      this.messages = [];
    }
  }

  saveChatHistory() {
    localStorage.setItem(this.getStorageKey(), JSON.stringify(this.chatHistory));
  }

  switchChat(chatId: number) {
    const chat = this.chatHistory.find(c => c.id === chatId);
    if (chat) {
      this.currentChatId = chatId;
      this.messages = chat.messages;
    }
  }

  // ----------------------------
  // NOUVEAU CHAT
  // ----------------------------
  startNewChat() {
    const newChat: Chat = {
      id: Date.now(),
      title: 'Nouvelle conversation',
      messages: []
    };
    this.chatHistory.unshift(newChat);
    this.currentChatId = newChat.id;
    this.messages = newChat.messages;
    this.saveChatHistory();
  }

  // ----------------------------
  // TITRE INTELLIGENT
  // ----------------------------
  private generateTitleFromMessage(message: string): string {
    const STOPWORDS = [
      "c'est","ce","ça","le","la","les","un","une","des","de","du",
      "au","aux","et","ou","dans","avec","pour","quoi","est","qui",
      "que","dont","je","tu","il","elle","on","nous","vous","ils","elles",
      "sur","à","par","en","aujourd'hui"
    ];

    let cleaned = message.toLowerCase()
      .replace(/[^\w\sàâçéèêëîïôûùüÿñæœ]/gi, ' ')
      .split(/\s+/)
      .filter(word => word && !STOPWORDS.includes(word));

    if (cleaned.length === 0) return "Nouvelle conversation";

    const salutations = ["bonjour","salut","hello","coucou"];
    const questions = ["c'est quoi", "comment", "pourquoi", "qu'est-ce", "qui", "où", "quand"];

    for (let s of salutations) if (message.toLowerCase().includes(s)) return "Salutations";
    for (let q of questions) {
      if (message.toLowerCase().includes(q)) {
        const idx = message.toLowerCase().indexOf(q) + q.length;
        const rest = message.slice(idx).trim().split(/\s+/).filter(w => !STOPWORDS.includes(w));
        const titleWords = rest.length ? rest.slice(0, 3) : cleaned.slice(0, 3);
        return titleWords.map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
      }
    }

    const title = cleaned.slice(0, 3).join(" ");
    return title.charAt(0).toUpperCase() + title.slice(1);
  }

  // ----------------------------
  // ENVOI DE MESSAGE
  // ----------------------------
  onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    this.selectedFile = input.files?.[0] ?? null;
  }

  sendMessage() {
    if (!this.userInput.trim() && !this.selectedFile) return;

    let content = this.userInput.trim();
    if (this.selectedFile) content += `\n[Fichier joint: ${this.selectedFile.name}]`;

    if (!this.currentChatId) this.startNewChat();
    const chat = this.chatHistory.find(c => c.id === this.currentChatId);
    if (!chat) return;

    if (chat.messages.length === 0) {
      chat.title = this.generateTitleFromMessage(this.userInput);
    }

    const userMsg: Message = { role: 'user', content, timestamp: new Date() };
    chat.messages.push(userMsg);
    this.messages = chat.messages;
    this.saveChatHistory();

    this.isLoading = true;

    const formData = new FormData();
    formData.append('query', this.userInput.trim());
    formData.append('conversation', JSON.stringify(chat.messages));
    if (this.selectedFile) formData.append('file', this.selectedFile);

    this.http.post('http://localhost:5010/chat_query', formData, { responseType: 'json' })
      .subscribe({
        next: (resJson: any) => {
          let fullResponse = resJson.response ?? '';

          if (resJson.provenance?.length) fullResponse += `\n\n📌 Provenance:\n- ${resJson.provenance.join('\n- ')}`;
          if (resJson.dates_found?.length) fullResponse += `\n\n📅 Dates trouvées:\n- ${resJson.dates_found.join('\n- ')}`;
          if (resJson.times_found?.length) fullResponse += `\n\n⏰ Heures trouvées:\n- ${resJson.times_found.join('\n- ')}`;
          if (resJson.warnings?.length)    fullResponse += `\n\n⚠️ Avertissements:\n- ${resJson.warnings.join('\n- ')}`;

          const msg: Message = {
            role: 'assistant',
            content: fullResponse,
            provenance: resJson.provenance,
            dates_found: resJson.dates_found,
            times_found: resJson.times_found,
            warnings: resJson.warnings,
            contact_hr: resJson.contact_hr ?? false,
            timestamp: new Date()
          };

 // ✅ Corrigé : on parcourt la liste renvoyée par le backend
if (resJson.matched_reports && resJson.matched_reports.length > 0) {
  // Ici je prends le premier fichier, mais tu peux gérer plusieurs liens
  const first = resJson.matched_reports[0];
  msg.fileName = first.filename;
  msg.downloadLink = first.download_link; // optionnel si tu veux un lien direct
  msg.fileType = "application/octet-stream";
}


          chat.messages.push(msg);
          this.saveChatHistory();

          this.userInput = '';
          this.selectedFile = null;
          this.isLoading = false;
        },
        error: err => {
          console.error(err);
          chat.messages.push({
            role: 'assistant',
            content: "❌ Erreur de communication avec l'assistant.",
            contact_hr: false,
            timestamp: new Date()
          });
          this.saveChatHistory();
          this.isLoading = false;
        }
      });
  }

  extractMainText(content: string): string {
    return content.split(/\n📌|\n📅|\n⏰|\n⚠️/)[0].trim();
  }

  contactAdvisor() {
    window.open('mailto:conseiller.rh@entreprise.com?subject=Demande%20d%27assistance', '_blank');
  }

  // ----------------------------
  // ✅ Téléchargement direct depuis /reports
  // ----------------------------
downloadFile(msg: Message) {
  const url = msg.downloadLink ?? `http://localhost:5010/reports/${encodeURIComponent(msg.fileName!)}`;
  const a = document.createElement('a');
  a.href = url;
  a.download = msg.fileName ?? 'document';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}
// ----------------------------
// SUPPRESSION D'UN CHAT
// ----------------------------
deleteChat(chatId: number, event: MouseEvent) {
  event.stopPropagation(); // empêche le click de switcher le chat

  const index = this.chatHistory.findIndex(c => c.id === chatId);
  if (index > -1) {
    this.chatHistory.splice(index, 1);
    this.saveChatHistory();

    // Si le chat supprimé était le chat courant, on switch sur le premier ou on vide
    if (this.currentChatId === chatId) {
      if (this.chatHistory.length) {
        this.switchChat(this.chatHistory[0].id);
      } else {
        this.currentChatId = null;
        this.messages = [];
      }
    }
  }
}


}