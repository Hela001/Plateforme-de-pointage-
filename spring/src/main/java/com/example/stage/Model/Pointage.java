package com.example.stage.Model;

import org.bson.types.Binary;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.Date;
import java.util.List;

@Document(collection = "pointage")
public class Pointage {

    @Id
    private String id;

    private String user_id;
    private String username;

    private LocalDate date_pointage;
    private LocalDateTime heure_pointage;

    private String statut;

    private Localisation localisation;

    private String adresse;
    private List<HeurePointage> arrivees;
    private List<HeurePointage> departs;

    public List<HeurePointage> getArrivees() {
        return arrivees;
    }

    public void setArrivees(List<HeurePointage> arrivees) {
        this.arrivees = arrivees;
    }

    public List<HeurePointage> getDeparts() {
        return departs;
    }

    public void setDeparts(List<HeurePointage> departs) {
        this.departs = departs;
    }

    @Field("image")
    private Binary image;

    @Field("role")
    private String role;

    // === Getters & Setters ===

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public String getUser_id() { return user_id; }
    public void setUser_id(String user_id) { this.user_id = user_id; }

    public String getUsername() { return username; }
    public void setUsername(String username) { this.username = username; }

    public LocalDate getDate_pointage() { return date_pointage; }
    public void setDate_pointage(LocalDate date_pointage) { this.date_pointage = date_pointage; }

    public LocalDateTime getHeure_pointage() { return heure_pointage; }
    public void setHeure_pointage(LocalDateTime heure_pointage) { this.heure_pointage = heure_pointage; }

    public String getStatut() { return statut; }
    public void setStatut(String statut) { this.statut = statut; }

    public Localisation getLocalisation() { return localisation; }
    public void setLocalisation(Localisation localisation) { this.localisation = localisation; }

    public String getAdresse() { return adresse; }
    public void setAdresse(String adresse) { this.adresse = adresse; }

    public Binary getImage() { return image; }
    public void setImage(Binary image) { this.image = image; }

    public String getRole() { return role; }
    public void setRole(String role) { this.role = role; }

    // === Localisation interne ===

    public static class Localisation {
        private double lat;
        private double lon;

        public double getLat() { return lat; }
        public void setLat(double lat) { this.lat = lat; }

        public double getLon() { return lon; }
        public void setLon(double lon) { this.lon = lon; }

    }


}
