# Saved Payment Methods Requirements

Customers can add, view, rename and remove saved payment methods. Customers must authenticate before managing payment methods. Card numbers must never be stored in plaintext; the system stores a provider token and masked metadata.

Adding a payment method depends on the Payment Tokenization API. Removing a payment method requires checking that it is not attached to an unsettled transaction. The Customer Profile Service owns the customer-to-payment-method relationship. Acceptance criterion: after successful addition, the customer sees card brand and last four digits.
