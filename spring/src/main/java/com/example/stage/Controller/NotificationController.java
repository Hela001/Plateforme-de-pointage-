package com.example.stage.Controller;


import com.example.stage.Model.Notification;
import com.example.stage.service.NotificationService;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/notifications")
@CrossOrigin(origins = "*")
public class NotificationController {

    private final NotificationService service;

    public NotificationController(NotificationService service) {
        this.service = service;
    }

    @GetMapping
    public List<Notification> getAllNotifications() {
        return service.getAll();
    }
    @GetMapping("/{id}")
    public Notification getNotificationById(@PathVariable String id) {
        return service.getById(id);
    }

}
