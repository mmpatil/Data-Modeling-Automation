'use strict';

module.exports = {
  up: (queryInterface, Sequelize) => {
    return queryInterface.createTable('ModelRunDetail',
     {
      Id: {
        allowNull: false,
        autoIncrement: true,
        primaryKey: true,
        type: Sequelize.INTEGER
      },
      RunId: {
        type: Sequelize.INTEGER,
        references: {
          model: 'RunDetail',
          key: 'Id'
        },
        onUpdate: 'cascade',
        onDelete: 'cascade'
      }
    });
  },

  down: (queryInterface, Sequelize) => {
    return queryInterface.dropTable('ModelRunDetail');
  }
};
