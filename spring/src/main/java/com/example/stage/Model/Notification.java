package com.example.stage.Model;

import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

import java.time.LocalDateTime;

@Document(collection = "notification")
public class Notification {

    @Id
    private String id;
    private String userId;
    private String username;
    private LocalDateTime date;
    private LocalDateTime heure;
    private String statut;
    private String message;
    private String role;
    private String adresse;

    public Notification() {}

    public Notification(String userId, String username, LocalDateTime date, LocalDateTime heure,
                        String statut, String message, String role, String adresse) {
        this.userId = userId;
        this.username = username;
        this.date = date;
        this.heure = heure;
        this.statut = statut;
        this.message = message;
        this.role = role;
        this.adresse = adresse;
    }

    // Getters et Setters
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getUserId() { return userId; }
    public void setUserId(String userId) { this.userId = userId; }
    public String getUsername() { return username; }
    public void setUsername(String username) { this.username = username; }
    public LocalDateTime getDate() { return date; }
    public void setDate(LocalDateTime date) { this.date = date; }
    public LocalDateTime getHeure() { return heure; }
    public void setHeure(LocalDateTime heure) { this.heure = heure; }
    public String getStatut() { return statut; }
    public void setStatut(String statut) { this.statut = statut; }
    public String getMessage() { return message; }
    public void setMessage(String message) { this.message = message; }
    public String getRole() { return role; }
    public void setRole(String role) { this.role = role; }
    public String getAdresse() { return adresse; }
    public void setAdresse(String adresse) { this.adresse = adresse; }
}
