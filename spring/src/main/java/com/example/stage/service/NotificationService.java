package com.example.stage.service;

import com.example.stage.Model.Notification;
import com.example.stage.Repository.NotificationRepository;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class NotificationService {

    private final NotificationRepository repository;

    public NotificationService(NotificationRepository repository) {
        this.repository = repository;
    }

    public List<Notification> getAll() {
        return repository.findAll();
    }
    public Notification getById(String id) {
        return repository.findById(id).orElse(null);
    }

}
