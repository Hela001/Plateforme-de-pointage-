import { Component, OnInit } from '@angular/core';
import { Fraude, FraudeService } from '../services/fraude-service';
import { CommonModule, DatePipe } from '@angular/common';
@Component({
  selector: 'app-fraude-list',
  standalone: true,
  imports: [CommonModule, DatePipe],
  templateUrl: './fraude-list.html',
  styleUrls: ['./fraude-list.scss']
})
export class FraudeListComponent implements OnInit {
  fraudes: Fraude[] = [];
  loading = false;
  errorMessage = '';

  constructor(private fraudeService: FraudeService) {}

  ngOnInit(): void {
    this.loading = true;
    this.fraudeService.getFraudes().subscribe({
      next: (data) => {
        this.fraudes = data;
        this.loading = false;
      },
      error: (err) => {
        this.errorMessage = 'Erreur lors du chargement des fraudes.';
        console.error(err);
        this.loading = false;
      }
    });
  }
}
