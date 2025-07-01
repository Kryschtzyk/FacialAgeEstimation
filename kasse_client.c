/*
 * WebSocket Client Beispiel für Kassensoftware
 * Kommuniziert mit dem Python Altersschätzungs-Service
 *
 * Kompilierung: gcc -o kasse_client kasse_client.c -lwebsockets -ljson-c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libwebsockets.h>
#include <json-c/json.h>

#define MAX_MESSAGE_SIZE 4096

typedef struct {
    struct lws *wsi;
    char message_buffer[MAX_MESSAGE_SIZE];
    int message_length;
    int age_result;
    char customer_message[256];
    char status[32];
} session_data;

// Callback-Funktionen für verschiedene Events
static int callback_age_estimation(struct lws *wsi, enum lws_callback_reasons reason,
                                  void *user, void *in, size_t len) {
    session_data *session = (session_data *)user;

    switch (reason) {
        case LWS_CALLBACK_CLIENT_ESTABLISHED:
            printf("[KASSE] Verbindung zum Altersschätzungs-Service hergestellt\n");

            // Kamera starten
            const char *start_camera = "{\"type\":\"start_camera\"}";
            lws_write(wsi, (unsigned char *)start_camera, strlen(start_camera), LWS_WRITE_TEXT);
            break;

        case LWS_CALLBACK_CLIENT_RECEIVE:
            {
                // JSON-Nachricht parsen
                json_object *root = json_tokener_parse((char *)in);
                if (!root) {
                    printf("[KASSE] Fehler beim JSON-Parsing\n");
                    break;
                }

                // Event-Typ ermitteln
                json_object *status_obj;
                if (json_object_object_get_ex(root, "status", &status_obj)) {
                    const char *status = json_object_get_string(status_obj);
                    strcpy(session->status, status);

                    if (strcmp(status, "complete") == 0) {
                        // Altersschätzung abgeschlossen
                        json_object *age_obj;
                        if (json_object_object_get_ex(root, "age", &age_obj)) {
                            session->age_result = json_object_get_int(age_obj);
                            printf("[KASSE] Alter geschätzt: %d Jahre\n", session->age_result);
                        }
                    }
                }

                // Kundennachricht extrahieren
                json_object *customer_msg_obj;
                if (json_object_object_get_ex(root, "customer_message", &customer_msg_obj)) {
                    const char *msg = json_object_get_string(customer_msg_obj);
                    strcpy(session->customer_message, msg);
                    printf("[KASSE] Kundennachricht: %s\n", msg);
                }

                json_object_put(root);
            }
            break;

        case LWS_CALLBACK_CLIENT_CONNECTION_ERROR:
            printf("[KASSE] Verbindungsfehler zum Altersschätzungs-Service\n");
            break;

        case LWS_CALLBACK_CLOSED:
            printf("[KASSE] Verbindung zum Altersschätzungs-Service geschlossen\n");
            break;

        default:
            break;
    }

    return 0;
}

// WebSocket-Protokoll definieren
static struct lws_protocols protocols[] = {
    {
        "age-estimation-protocol",
        callback_age_estimation,
        sizeof(session_data),
        MAX_MESSAGE_SIZE,
    },
    { NULL, NULL, 0, 0 } // Terminator
};

// Hauptfunktion für Altersschätzung
int start_age_estimation(session_data *session) {
    // Analyse starten
    char start_message[256];
    snprintf(start_message, sizeof(start_message),
             "{\"type\":\"start_analysis\", \"session_id\":\"kasse_%ld\"}",
             time(NULL));

    if (session->wsi) {
        lws_write(session->wsi, (unsigned char *)start_message,
                 strlen(start_message), LWS_WRITE_TEXT);
        printf("[KASSE] Altersanalyse gestartet\n");
        return 0;
    }
    return -1;
}

// Session zurücksetzen
int reset_age_estimation(session_data *session) {
    char reset_message[256];
    snprintf(reset_message, sizeof(reset_message),
             "{\"type\":\"reset_session\", \"session_id\":\"kasse_%ld\"}",
             time(NULL));

    if (session->wsi) {
        lws_write(session->wsi, (unsigned char *)reset_message,
                 strlen(reset_message), LWS_WRITE_TEXT);
        session->age_result = 0;
        strcpy(session->customer_message, "");
        strcpy(session->status, "");
        printf("[KASSE] Session zurückgesetzt\n");
        return 0;
    }
    return -1;
}

// Hauptfunktion - Beispiel für Integration
int main() {
    struct lws_context_creation_info info;
    struct lws_context *context;
    struct lws_client_connect_info connect_info;
    session_data session = {0};

    // WebSocket-Kontext initialisieren
    memset(&info, 0, sizeof(info));
    info.port = CONTEXT_PORT_NO_LISTEN;
    info.protocols = protocols;
    info.gid = -1;
    info.uid = -1;

    context = lws_create_context(&info);
    if (!context) {
        printf("[KASSE] Fehler beim Erstellen des WebSocket-Kontexts\n");
        return -1;
    }

    // Verbindung zum Altersschätzungs-Service herstellen
    memset(&connect_info, 0, sizeof(connect_info));
    connect_info.context = context;
    connect_info.address = "localhost";  // Docker-Container IP
    connect_info.port = 5001;
    connect_info.path = "/socket.io/?EIO=4&transport=websocket";
    connect_info.host = connect_info.address;
    connect_info.origin = connect_info.address;
    connect_info.protocol = protocols[0].name;
    connect_info.userdata = &session;

    session.wsi = lws_client_connect_via_info(&connect_info);
    if (!session.wsi) {
        printf("[KASSE] Fehler beim Verbinden zum Service\n");
        lws_context_destroy(context);
        return -1;
    }

    printf("[KASSE] Altersschätzungs-Service gestartet\n");
    printf("[KASSE] Verfügbare Befehle:\n");
    printf("  1 - Altersanalyse starten\n");
    printf("  2 - Session zurücksetzen\n");
    printf("  q - Beenden\n");

    // Hauptschleife
    char command;
    while (1) {
        // WebSocket-Events verarbeiten
        lws_service(context, 50);

        // Benutzereingabe prüfen (non-blocking)
        if (scanf(" %c", &command) == 1) {
            switch (command) {
                case '1':
                    start_age_estimation(&session);
                    break;
                case '2':
                    reset_age_estimation(&session);
                    break;
                case 'q':
                    goto cleanup;
                default:
                    printf("[KASSE] Unbekannter Befehl\n");
                    break;
            }
        }

        // Status anzeigen
        if (strlen(session.customer_message) > 0) {
            printf("[DISPLAY] %s\n", session.customer_message);
        }

        if (session.age_result > 0) {
            printf("[RESULT] Geschätztes Alter: %d Jahre\n", session.age_result);
            // Hier würden Sie das Alter in Ihrer Kassensoftware verarbeiten
        }
    }

cleanup:
    lws_context_destroy(context);
    printf("[KASSE] Service beendet\n");
    return 0;
}

/*
 * Integration in bestehende Kassensoftware:
 *
 * 1. Diese Funktionen in Ihre C-Anwendung einbinden
 * 2. WebSocket-Verbindung beim Kassenstart initialisieren
 * 3. Bei Altersverifikation start_age_estimation() aufrufen
 * 4. customer_message für Display verwenden
 * 5. age_result für Geschäftslogik nutzen
 *
 * Beispiel-Integration:
 *
 * void check_customer_age() {
 *     start_age_estimation(&global_session);
 *
 *     while (strcmp(global_session.status, "complete") != 0 &&
 *            strcmp(global_session.status, "error") != 0) {
 *         lws_service(global_context, 100);
 *         update_customer_display(global_session.customer_message);
 *     }
 *
 *     if (global_session.age_result >= 18) {
 *         allow_purchase();
 *     } else {
 *         deny_purchase();
 *     }
 * }
 */
